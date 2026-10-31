# GOLD: Globally Optimal Supervised Discretisation Algorithm
# Univariate exact dynamic programming using penalised leave-one-out Brier loss.
# For each feature: minimise sum_b LOO_Brier(b) + (m - 1) * log(N)/2.
# Requires numeric predictors and categorical class in the last column.
# This file defines functions only; it does not run experiments when sourced.
# The public interface is gold(data).  No third-party dependencies.

# Multiclass sum-of-squares Brier score for an entire bin, efficiently O(K).
gold_bin_loss <- function(counts) {
  alpha <- 0.5  # Fixed symmetric Jeffreys-type pseudo-count
  counts <- as.numeric(counts)
  K <- length(counts); n <- sum(counts)
  if (n < 1L || K < 2L) return(0)
  d <- n - 1 + K*alpha
  a <- counts + alpha
  # sum_i ||q_{-i} - onehot(y_i)||^2
  n - 2*sum(counts*(a-1))/d +
    (n*sum(a*a) - 2*sum(counts*a) + n)/(d*d)
}

gold_discretize <- function(x, y, min_bin_size = 5L) {
  stopifnot(is.numeric(x), length(x) == length(y),
            is.finite(min_bin_size), min_bin_size >= 1)
  n_original <- length(x)
  ok <- is.finite(x) & !is.na(y)
  x <- x[ok]; y <- droplevels(as.factor(y[ok]))
  N <- length(x)
  if (!N) stop('No complete observations.')
  if (N < min_bin_size) stop('Fewer complete observations than min_bin_size.')
  K <- nlevels(y)
  if (K < 2L) stop('Need at least two observed outcome classes.')
  ord <- order(x)
  xs <- x[ord]; ys <- as.integer(y[ord]); vals <- unique(xs)
  G <- length(vals)
  gi <- match(xs, vals)
  counts_gk <- matrix(0, nrow=G, ncol=K)
  for (i in seq_along(gi)) counts_gk[gi[i], ys[i]] <- counts_gk[gi[i], ys[i]] + 1
  pref <- rbind(rep(0, K), apply(counts_gk, 2, cumsum))
  pref_n <- c(0, cumsum(rowSums(counts_gk)))
  dp <- rep(-Inf, G+1L); dp[1] <- 0
  back <- rep(NA_integer_, G+1L)
  penalty <- 0.5*log(N)  # Fixed per-cut penalty
  for (i in seq_len(G)) {
    for (j in seq_len(i)) {
      nbin <- pref_n[i+1] - pref_n[j]
      if (nbin < min_bin_size || !is.finite(dp[j])) next
      counts <- pref[i+1,] - pref[j,]
      score <- dp[j] - gold_bin_loss(counts) - penalty
      if (score > dp[i+1]) {
        dp[i+1] <- score
        back[i+1] <- j
      }
    }
  }
  if (!is.finite(dp[G+1L])) stop('No feasible partition found.')
  starts <- integer(0); i <- G
  while(i > 0L) {
    j <- back[i+1L]
    starts <- c(j, starts)
    i <- j-1L
  }
  if (length(starts) == 1L) {
    cuts <- numeric(0)
  } else {
    k <- starts[-1L] - 1L
    # numerically safer midpoint than (a+b)/2
    cuts <- vals[k] + (vals[k+1L]-vals[k])/2
  }
  breaks <- c(-Inf, cuts, Inf)
  m <- length(breaks)-1L
  bins <- cut(x, breaks=breaks, include.lowest=TRUE, right=TRUE)
  # DP subtracted per-bin penalty; add one back to obtain (m-1) cut penalty
  objective <- dp[G+1L] + penalty
  list(breaks=breaks, cutpoints=cuts, bins=bins, n_bins=m,
       objective=objective, penalised_loss=-objective,
       loo_brier_sum=-objective-penalty*(m-1),
       loo_brier_mean=(-objective-penalty*(m-1))/N,
       alpha=0.5, penalty_per_cut=penalty, n_used=N, n_dropped=n_original-N,
       dp_score=dp[G+1L])
}

# Data frame input: numeric predictors, categorical response in last column.
# Fit to training fold only, then apply stored cutpoints to test fold.
gold <- function(data) {
  class_col <- ncol(data)
  if (class_col < 2L) stop('Need features and class column.')
  if (anyNA(data)) stop('GOLD requires complete data; preprocess missing values before fitting.')
  if (any(!vapply(data[-class_col], is.numeric, logical(1))))
    stop('GOLD requires numeric predictor columns.')
  if (any(!vapply(data[-class_col], function(z) all(is.finite(z)), logical(1))))
    stop('GOLD requires finite predictor values.')
  min_bin_size <- max(5L, floor(nrow(data)/20L))
  cutp <- vector('list', class_col-1L)
  Disc.data <- data
  bcounts <- numeric(0)
  mean_scores <- numeric(0)
  for (i in seq_len(class_col-1L)) {
    res <- gold_discretize(data[[i]], data[[class_col]],
                               min_bin_size=min_bin_size)
    cutp[[i]] <- if (length(res$cutpoints)) res$cutpoints else 'All'
    Disc.data[[i]] <- as.integer(cut(data[[i]], res$breaks,
                                     include.lowest=TRUE, right=TRUE))
    bcounts <- c(bcounts, res$n_bins)
    mean_scores <- c(mean_scores, res$loo_brier_mean)
  }
  list(cutp=cutp, Disc.data=Disc.data,
       avg_bins=if(length(bcounts)) mean(bcounts) else NA_real_,
       total_bins=sum(bcounts), mean_loo_brier=if(length(mean_scores)) mean(mean_scores) else NA_real_)
}

# Apply training-derived cutpoints to new data with the same predictor columns.
# The response column, when present, is left unchanged.
gold_transform <- function(data, cutp) {
  data <- as.data.frame(data)
  p <- length(cutp)
  if (ncol(data) < p) stop('Insufficient predictor columns.')
  for (j in seq_len(p)) {
    cp <- cutp[[j]]
    if (is.null(cp)) stop('Missing cutpoints for predictor ', j)
    breaks <- if (identical(cp, 'All') || length(cp) == 0L) {
      c(-Inf, Inf)
    } else {
      c(-Inf, as.numeric(cp), Inf)
    }
    data[[j]] <- as.integer(cut(data[[j]], breaks = breaks,
                                include.lowest = TRUE, right = TRUE))
  }
  data
}
