# =============================================================================
# GOLD: Globally Optimal Log-likelihood Discretisation
# =============================================================================
#
# Core objective: maximise BIC-adjusted McFadden R² over all partitions
#
#   R²_adj(S) = 1 - [ ℓ(S) - m·λ·(K-1)·log(N)/2 ] / ℓ₀
#
# where:
#   ℓ(S)  = saturated multinomial log-likelihood of scheme S
#   ℓ₀    = null log-likelihood (global class proportions, no feature)
#   m     = number of bins
#   K     = number of classes
#   N     = sample size
#   λ     = penalty scaling parameter (λ=1 → standard BIC-adjusted R²)
#
# Since ℓ₀ < 0 is fixed, maximising R²_adj is equivalent to maximising
# the penalised log-likelihood Q(S) = ℓ(S) - m·λ·(K-1)·log(N)/2,
# which the dynamic programming algorithm solves exactly in O(G²K) time.
#
# This makes GOLD the first discretisation algorithm that is provably
# globally optimal under a BIC-adjusted McFadden R² criterion.
#
# COMPARISON 1: GOLD vs 7 competing discretisers
#   Classifiers : NaiveBayes, LogisticReg, XGBoost, OneR
#   Statistics  : Friedman + Holm-corrected Wilcoxon
#
# COMPARISON 2: GOLD vs None (no discretisation) — accuracy only
#   Classifiers : GaussianNB, LogisticReg, XGBoost
#   Statistics  : 3 independent Wilcoxon tests
#
# Datasets : 22 (mlbench + base R)
#
# Required packages:
#   install.packages(c("discretization", "nnet", "xgboost", "mlbench"))
# =============================================================================

need    <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop(sprintf("Package '%s' not installed. Run: install.packages('%s')", pkg, pkg),
         call. = FALSE)
}
has_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)

# =============================================================================
# PART 1 — R² helpers
# =============================================================================

# Null log-likelihood: global class proportions, no feature information
loglik_null <- function(y_factor) {
  tab <- table(as.factor(y_factor));  N <- sum(tab)
  p   <- as.numeric(tab) / N
  sum(as.numeric(tab) * log(p))
}

# Saturated bin log-likelihood from final breaks (for reporting)
loglik_bins <- function(x, y, breaks) {
  y     <- as.factor(y)
  bins  <- cut(x, breaks = breaks, include.lowest = TRUE, right = TRUE)
  lev_y <- levels(y);  ll <- 0
  for (b in levels(bins)) {
    idx <- bins == b;  tab <- table(y[idx]);  n <- sum(tab)
    if (n == 0) next
    cnt <- rep(0, length(lev_y));  names(cnt) <- lev_y
    cnt[names(tab)] <- as.integer(tab);  pos <- cnt > 0
    ll <- ll + sum(cnt[pos] * log(cnt[pos] / n))
  }
  ll
}

# BIC-adjusted McFadden R² (the quantity GOLD actually maximises)
#   R²_adj = 1 - [ ℓ(S) - m·λ·(K-1)·log(N)/2 ] / ℓ₀
# Also returns raw R² = 1 - ℓ(S)/ℓ₀ for comparison/reporting
r2_gold <- function(ll_model, ll_null, n_bins, K, N, lambda = 1) {
  if (!is.finite(ll_model) || !is.finite(ll_null) ||
      ll_null == 0 || N <= 0)
    return(list(R2_raw = NA_real_, R2_adj = NA_real_))

  bic_penalty <- n_bins * lambda * (K - 1) * log(N) / 2
  list(
    R2_raw = 1 - ll_model / ll_null,                          # unadjusted
    R2_adj = 1 - (ll_model - bic_penalty) / ll_null           # BIC-adjusted
  )
}

# =============================================================================
# PART 2 — GOLD discretisation
# =============================================================================

gold_discretize <- function(x, y, min_bin_size = 5, lambda = 1) {
  stopifnot(is.numeric(x), length(x) == length(y))
  n0 <- length(x)
  ok <- is.finite(x) & !is.na(y);  x <- x[ok];  y <- y[ok]
  if (!length(x)) stop("No complete (x,y) pairs.")

  y  <- as.factor(y);  K <- nlevels(y);  N <- length(y)
  ll_null <- loglik_null(y)

  # BIC penalty per bin — the adjustment term in BIC-adjusted R²
  # Maximising R²_adj ≡ maximising Q(S) = ℓ(S) - m·λ·(K-1)·log(N)/2
  penalty_per_bin <- lambda * (K - 1) * log(N) / 2

  ord <- order(x);  xs <- x[ord];  ys <- y[ord]

  if (length(unique(xs)) == 1L) {
    breaks <- c(-Inf, Inf)
    ll_m   <- loglik_bins(x, y, breaks)
    r2     <- r2_gold(ll_m, ll_null, 1L, K, N, lambda)
    return(list(
      breaks = breaks, cutpoints = numeric(0),
      bins   = cut(x, breaks, include.lowest = TRUE),
      ll_model = ll_m, ll_null = ll_null,
      R2_raw = r2$R2_raw, R2_adj = r2$R2_adj,
      n_bins = 1L, n_used = N, n_dropped = n0 - N
    ))
  }

  # compress equal-x runs
  r       <- rle(xs);  G <- length(r$values);  v <- r$values
  g_end   <- cumsum(r$lengths)
  g_start <- c(1L, head(g_end, -1L) + 1L)
  n_g     <- r$lengths;  lev <- levels(ys)

  counts_gk <- matrix(0L, G, K, dimnames = list(NULL, lev))
  for (g in seq_len(G)) {
    tab <- table(ys[g_start[g]:g_end[g]])
    counts_gk[g, names(tab)] <- as.integer(tab)
  }
  pref_counts <- rbind(rep(0L, K), apply(counts_gk, 2, cumsum))
  pref_n      <- c(0L, cumsum(n_g))

  # interval log-likelihood (numerator of R²_adj contribution for this bin)
  interval_ll <- function(j, i) {
    n <- pref_n[i + 1L] - pref_n[j];  if (n < min_bin_size) return(-Inf)
    cnt <- pref_counts[i + 1L, ] - pref_counts[j, ];  pos <- cnt > 0
    sum(cnt[pos] * log(cnt[pos] / n))
  }

  # DP: maximise Q(S) = ℓ(S) - m·penalty_per_bin
  # Equivalent to maximising BIC-adjusted McFadden R²
  dp <- rep(-Inf, G + 1L);  last <- rep(NA_integer_, G + 1L);  dp[1L] <- 0
  for (i in seq_len(G)) {
    best_val <- -Inf;  best_j <- NA_integer_
    for (j in seq_len(i)) {
      prev <- dp[j];  if (!is.finite(prev)) next
      ll   <- interval_ll(j, i);  if (!is.finite(ll)) next
      cand <- prev + ll - penalty_per_bin
      if (cand > best_val) { best_val <- cand;  best_j <- j }
    }
    dp[i + 1L] <- best_val;  last[i + 1L] <- best_j
  }
  if (!is.finite(dp[G + 1L])) stop("DP failed. Try reducing min_bin_size.")

  # traceback
  starts <- integer(0);  i <- G
  while (i > 0L) { j <- last[i + 1L];  starts <- c(j, starts);  i <- j - 1L }

  cutpoints <- if (length(starts) <= 1L) numeric(0) else {
    k <- starts[-1L] - 1L;  (v[k] + v[k + 1L]) / 2
  }

  breaks   <- c(-Inf, cutpoints, Inf)
  bins     <- cut(x, breaks = breaks, include.lowest = TRUE, right = TRUE)
  ll_model <- loglik_bins(x, y, breaks)
  m        <- length(breaks) - 1L
  r2       <- r2_gold(ll_model, ll_null, m, K, N, lambda)

  list(
    breaks    = breaks,
    cutpoints = cutpoints,
    bins      = bins,
    ll_model  = ll_model,
    ll_null   = ll_null,
    R2_raw    = r2$R2_raw,    # unadjusted McFadden R² (post-hoc diagnostic)
    R2_adj    = r2$R2_adj,    # BIC-adjusted McFadden R² (what GOLD maximises)
    n_bins    = m,
    n_used    = N,
    n_dropped = n0 - N
  )
}

# disc.Topdown-compatible wrapper
gold <- function(data, lambda = 1) {
  class_col    <- ncol(data)
  min_bin_size <- max(5L, floor(nrow(data) / 20L))
  feature_cols <- seq_len(class_col - 1L)
  cutp         <- vector("list", length(feature_cols))
  Disc.data    <- data
  bin_counts   <- integer(0)
  r2_adj_vals  <- numeric(0)

  for (i in feature_cols) {
    col <- data[[i]];  if (!is.numeric(col)) next
    res        <- gold_discretize(col, data[[class_col]],
                                  min_bin_size = min_bin_size,
                                  lambda = lambda)
    cutp[[i]]      <- if (length(res$cutpoints) == 0L) "All" else res$cutpoints
    Disc.data[[i]] <- as.integer(cut(col, breaks = res$breaks,
                                     include.lowest = TRUE, right = TRUE))
    bin_counts  <- c(bin_counts,  res$n_bins)
    r2_adj_vals <- c(r2_adj_vals, res$R2_adj)
  }

  list(cutp       = cutp,
       Disc.data  = Disc.data,
       avg_bins   = if (length(bin_counts))  round(mean(bin_counts),  2) else NA_real_,
       total_bins = sum(bin_counts),
       mean_R2adj = if (length(r2_adj_vals)) round(mean(r2_adj_vals, na.rm=TRUE), 4)
       else NA_real_)
}

# No-discretisation passthrough
no_disc <- function(data, ...) {
  class_col <- ncol(data)
  cutp <- lapply(seq_len(class_col - 1L), function(i) "All")
  list(cutp = cutp, Disc.data = data,
       avg_bins = NA_real_, total_bins = NA_integer_, mean_R2adj = NA_real_)
}

# =============================================================================
# PART 3 — Classifiers
# =============================================================================

# ---- Discrete Naive Bayes --------------------------------------------------
nb_train <- function(Disc.data, alpha = 1) {
  class_col <- ncol(Disc.data);  y <- as.factor(Disc.data[[class_col]])
  classes <- levels(y);  n_cls <- length(classes)
  log_prior <- log((as.numeric(table(y)) + alpha) / (length(y) + alpha * n_cls))
  names(log_prior) <- classes
  cond <- lapply(seq_len(class_col - 1L), function(i) {
    col <- Disc.data[[i]];  bins <- sort(unique(col));  nb <- length(bins)
    log_lk <- matrix(0, nb, n_cls, dimnames = list(bins, classes))
    for (cls in classes) {
      idx <- y == cls
      cnt <- as.numeric(table(factor(col[idx], levels = bins)))
      log_lk[, cls] <- log((cnt + alpha) / (sum(idx) + alpha * nb))
    }
    list(bins = bins, log_lk = log_lk)
  })
  list(log_prior=log_prior, cond=cond, classes=classes,
       class_col=class_col, type="nb")
}
nb_predict <- function(model, Disc.data) {
  n <- nrow(Disc.data)
  log_post <- matrix(rep(model$log_prior, each=n), n, length(model$classes),
                     dimnames=list(NULL, model$classes))
  for (i in seq_len(model$class_col - 1L)) {
    ci <- model$cond[[i]];  if (is.null(ci)) next
    col <- Disc.data[[i]]
    for (j in seq_len(n)) {
      b <- as.character(col[j])
      if (b %in% rownames(ci$log_lk))
        log_post[j,] <- log_post[j,] + ci$log_lk[b,]
    }
  }
  model$classes[apply(log_post, 1, which.max)]
}

# ---- Gaussian Naive Bayes (continuous baseline) ----------------------------
gnb_train <- function(data) {
  class_col <- ncol(data);  y <- as.factor(data[[class_col]])
  classes <- levels(y);  n_cls <- length(classes)
  log_prior <- log(as.numeric(table(y)) / length(y))
  names(log_prior) <- classes
  params <- lapply(seq_len(class_col - 1L), function(i) {
    col <- as.numeric(data[[i]])
    setNames(lapply(classes, function(cls) {
      vals <- col[y == cls]
      list(mu=mean(vals, na.rm=TRUE), sigma=max(sd(vals, na.rm=TRUE), 1e-9))
    }), classes)
  })
  list(log_prior=log_prior, params=params, classes=classes,
       class_col=class_col, type="gnb")
}
gnb_predict <- function(model, data) {
  n <- nrow(data)
  log_post <- matrix(rep(model$log_prior, each=n), n, length(model$classes),
                     dimnames=list(NULL, model$classes))
  for (i in seq_len(model$class_col - 1L)) {
    col <- as.numeric(data[[i]])
    for (cls in model$classes) {
      p <- model$params[[i]][[cls]]
      log_post[,cls] <- log_post[,cls] + dnorm(col, p$mu, p$sigma, log=TRUE)
    }
  }
  model$classes[apply(log_post, 1, which.max)]
}

# ---- Logistic Regression ---------------------------------------------------
lr_train <- function(Disc.data, continuous = FALSE) {
  need("nnet");  class_col <- ncol(Disc.data);  df <- Disc.data
  df[[class_col]] <- as.factor(df[[class_col]])
  if (!continuous)
    for (i in seq_len(class_col - 1L)) df[[i]] <- as.factor(df[[i]])
  model <- nnet::multinom(as.formula(paste(names(df)[class_col], "~ .")),
                          data=df, trace=FALSE, maxit=200)
  list(model=model, class_col=class_col, continuous=continuous, type="lr")
}
lr_predict <- function(fit, Disc.data) {
  df <- Disc.data
  if (!fit$continuous)
    for (i in seq_len(fit$class_col - 1L)) df[[i]] <- as.factor(df[[i]])
  as.character(predict(fit$model, newdata=df))
}

# ---- XGBoost ---------------------------------------------------------------
xgb_train <- function(Disc.data) {
  need("xgboost");  class_col <- ncol(Disc.data)
  y_raw <- as.factor(Disc.data[[class_col]]);  y_int <- as.integer(y_raw) - 1L
  X <- as.matrix(Disc.data[,-class_col,drop=FALSE])
  storage.mode(X) <- "double";  n_cls <- nlevels(y_raw)
  params <- Filter(Negate(is.null), list(
    objective   = if (n_cls==2) "binary:logistic" else "multi:softmax",
    num_class   = if (n_cls>2) n_cls else NULL,
    eta=0.1, max_depth=4, nthread=1,
    eval_metric = if (n_cls==2) "error" else "merror"))
  model <- xgboost::xgb.train(params,
                              xgboost::xgb.DMatrix(X, label=y_int), nrounds=50, verbose=0)
  list(model=model, classes=levels(y_raw),
       class_col=class_col, n_cls=n_cls, type="xgb")
}
xgb_predict <- function(fit, Disc.data) {
  X <- as.matrix(Disc.data[,-fit$class_col,drop=FALSE])
  storage.mode(X) <- "double"
  raw <- predict(fit$model, xgboost::xgb.DMatrix(X))
  idx <- if (fit$n_cls==2) ifelse(raw>0.5,2L,1L) else as.integer(raw)+1L
  fit$classes[idx]
}

# ---- OneR ------------------------------------------------------------------
oner_train <- function(Disc.data) {
  class_col <- ncol(Disc.data);  y <- as.factor(Disc.data[[class_col]])
  classes <- levels(y);  best_feat <- NULL;  best_rules <- NULL;  best_err <- Inf
  for (i in seq_len(class_col - 1L)) {
    bins <- Disc.data[[i]];  u <- sort(unique(bins))
    rules <- setNames(vapply(u, function(b) {
      sub <- y[bins==b]
      if (!length(sub)) classes[1L] else names(which.max(table(sub)))
    }, character(1)), as.character(u))
    err <- mean(rules[as.character(bins)] != as.character(y))
    if (err < best_err) { best_err <- err;  best_feat <- i;  best_rules <- rules }
  }
  list(feat=best_feat, rules=best_rules, default=classes[1L],
       class_col=class_col, type="oner")
}
oner_predict <- function(fit, Disc.data) {
  col <- as.character(Disc.data[[fit$feat]])
  as.character(ifelse(col %in% names(fit$rules), fit$rules[col], fit$default))
}

# ---- dispatch --------------------------------------------------------------
clf_train <- function(type, data, continuous=FALSE) {
  switch(type, nb=nb_train(data), gnb=gnb_train(data),
         lr=lr_train(data, continuous=continuous),
         xgb=xgb_train(data), oner=oner_train(data),
         stop("Unknown: ", type))
}
clf_predict <- function(fit, data) {
  switch(fit$type, nb=nb_predict(fit,data), gnb=gnb_predict(fit,data),
         lr=lr_predict(fit,data), xgb=xgb_predict(fit,data),
         oner=oner_predict(fit,data))
}

# =============================================================================
# PART 3b — Probability predictions (for AUC)
# Each function returns an n x K matrix of class probabilities.
# Column names must match class labels.
# =============================================================================

nb_predict_prob <- function(model, Disc.data) {
  n <- nrow(Disc.data);  K <- length(model$classes)
  log_post <- matrix(rep(model$log_prior, each=n), n, K,
                     dimnames=list(NULL, model$classes))
  for (i in seq_len(model$class_col - 1L)) {
    ci <- model$cond[[i]];  if (is.null(ci)) next
    col <- Disc.data[[i]]
    for (j in seq_len(n)) {
      b <- as.character(col[j])
      if (b %in% rownames(ci$log_lk))
        log_post[j,] <- log_post[j,] + ci$log_lk[b,]
    }
  }
  # softmax to convert log-posteriors to probabilities
  log_post <- log_post - apply(log_post, 1, max)
  prob     <- exp(log_post)
  prob / rowSums(prob)
}

gnb_predict_prob <- function(model, data) {
  n <- nrow(data);  K <- length(model$classes)
  log_post <- matrix(rep(model$log_prior, each=n), n, K,
                     dimnames=list(NULL, model$classes))
  for (i in seq_len(model$class_col - 1L)) {
    col <- as.numeric(data[[i]])
    for (cls in model$classes) {
      p <- model$params[[i]][[cls]]
      log_post[,cls] <- log_post[,cls] + dnorm(col, p$mu, p$sigma, log=TRUE)
    }
  }
  log_post <- log_post - apply(log_post, 1, max)
  prob     <- exp(log_post)
  prob / rowSums(prob)
}

lr_predict_prob <- function(fit, Disc.data) {
  df <- Disc.data
  if (!fit$continuous)
    for (i in seq_len(fit$class_col - 1L)) df[[i]] <- as.factor(df[[i]])
  prob <- tryCatch(predict(fit$model, newdata=df, type="probs"),
                   error=function(e) NULL)
  if (is.null(prob)) return(NULL)
  # binary case: nnet returns a vector, not a matrix
  if (is.vector(prob)) {
    prob <- cbind(1 - prob, prob)
    colnames(prob) <- fit$model$lev
  }
  as.matrix(prob)
}

xgb_predict_prob <- function(fit, Disc.data) {
  X <- as.matrix(Disc.data[,-fit$class_col, drop=FALSE])
  storage.mode(X) <- "double"
  raw <- predict(fit$model, xgboost::xgb.DMatrix(X))
  if (fit$n_cls == 2L) {
    prob <- cbind(1 - raw, raw)
  } else {
    prob <- matrix(raw, ncol=fit$n_cls, byrow=TRUE)
  }
  colnames(prob) <- fit$classes
  prob
}

oner_predict_prob <- function(fit, Disc.data) {
  # OneR has no probabilities — return 0/1 hard indicator
  preds       <- oner_predict(fit, Disc.data)
  all_classes <- sort(unique(c(names(fit$rules), fit$default)))
  prob        <- matrix(0, nrow=length(preds), ncol=length(all_classes))
  colnames(prob) <- all_classes
  for (i in seq_along(preds)) prob[i, preds[i]] <- 1
  prob
}

# ── probability dispatch ──────────────────────────────────────────────────────
clf_predict_prob <- function(fit, data) {
  switch(fit$type,
         nb   = nb_predict_prob(fit, data),
         gnb  = gnb_predict_prob(fit, data),
         lr   = lr_predict_prob(fit, data),
         xgb  = xgb_predict_prob(fit, data),
         oner = oner_predict_prob(fit, data),
         NULL
  )
}

# ── AUC via pROC — Hand & Till (2001) multiclass estimator ───────────────────
# prob_mat : n x K matrix with colnames = class labels
# labels   : character vector of true class labels (length n)
compute_auc <- function(prob_mat, labels) {
  if (!has_pkg("pROC")) {
    message("pROC not installed — AUC not computed. Run: install.packages('pROC')")
    return(NA_real_)
  }
  tryCatch({
    rc <- pROC::multiclass.roc(response  = labels,
                               predictor = prob_mat,
                               quiet     = TRUE)
    as.numeric(rc$auc)
  }, error = function(e) NA_real_)
}

# =============================================================================
# PART 4 — Cross-validation core
# =============================================================================

apply_breaks <- function(data, cutp, class_col) {
  out <- data
  for (i in seq_len(class_col - 1L)) {
    col <- data[[i]];  if (!is.numeric(col)) next
    cp  <- cutp[[i]]
    br  <- if (identical(cp,"All") || length(cp)==0) c(-Inf,Inf)
    else c(-Inf, as.numeric(cp), Inf)
    out[[i]] <- as.integer(cut(col, breaks=br,
                               include.lowest=TRUE, right=TRUE))
  }
  out
}

count_bins <- function(cutp) {
  sapply(cutp, function(cp) {
    if (is.null(cp))          return(NA_integer_)
    if (identical(cp,"All"))  return(1L)
    length(cp) + 1L
  })
}

cv_run <- function(data, disc_fn, clf_types, k=10, seed=42,
                   continuous=FALSE, track_bins=TRUE, ...) {
  set.seed(seed)
  n         <- nrow(data);  folds <- sample(rep(seq_len(k), length.out=n))
  class_col <- ncol(data);  y <- as.factor(data[[class_col]])
  classes   <- levels(y)
  correct   <- setNames(integer(length(clf_types)), clf_types)
  disc_time <- 0
  clf_time  <- setNames(numeric(length(clf_types)), clf_types)
  all_bins  <- list()
  all_r2adj <- numeric(0)

  # accumulate out-of-fold probability predictions for AUC
  fold_probs <- setNames(
    lapply(clf_types, function(ct)
      list(probs=NULL, labels=character(0))),
    clf_types)

  for (fold in seq_len(k)) {
    train  <- data[folds != fold,,drop=FALSE]
    test   <- data[folds == fold,,drop=FALSE]
    y_test <- as.character(y[folds==fold])

    t0         <- proc.time()["elapsed"]
    disc_train <- tryCatch(disc_fn(train,...), error=function(e) NULL)
    disc_time  <- disc_time + (proc.time()["elapsed"] - t0)
    if (is.null(disc_train)) next

    if (track_bins) {
      all_bins[[fold]] <- count_bins(disc_train$cutp)
      if (!is.null(disc_train$mean_R2adj) && !is.na(disc_train$mean_R2adj))
        all_r2adj <- c(all_r2adj, disc_train$mean_R2adj)
    }

    disc_test <- if (continuous) test else
      apply_breaks(test, disc_train$cutp, class_col)

    for (ct in clf_types) {
      t1  <- proc.time()["elapsed"]
      fit <- tryCatch(clf_train(ct, disc_train$Disc.data, continuous=continuous),
                      error=function(e) NULL)
      if (is.null(fit)) next

      # ── accuracy ──────────────────────────────────────────────────────────
      preds <- tryCatch(clf_predict(fit, disc_test), error=function(e) NULL)
      clf_time[ct] <- clf_time[ct] + (proc.time()["elapsed"] - t1)
      if (!is.null(preds))
        correct[ct] <- correct[ct] + sum(preds == y_test)

      # ── probabilities for AUC (accumulate across folds) ───────────────────
      probs <- tryCatch(clf_predict_prob(fit, disc_test), error=function(e) NULL)
      if (!is.null(probs)) {
        # align columns to global class set in case a fold is missing a class
        aligned <- matrix(0, nrow=nrow(probs), ncol=length(classes))
        colnames(aligned) <- classes
        common <- intersect(colnames(probs), classes)
        aligned[, common] <- probs[, common, drop=FALSE]
        fold_probs[[ct]]$probs  <- rbind(fold_probs[[ct]]$probs, aligned)
        fold_probs[[ct]]$labels <- c(fold_probs[[ct]]$labels, y_test)
      }
    }
  }

  # ── AUC from full out-of-fold prediction matrix ───────────────────────────
  auc <- setNames(vapply(clf_types, function(ct) {
    fp <- fold_probs[[ct]]
    if (is.null(fp$probs) || length(fp$labels) == 0L) return(NA_real_)
    compute_auc(fp$probs, fp$labels)
  }, numeric(1)), clf_types)

  bin_mat    <- do.call(rbind, all_bins)
  avg_bins   <- if (!track_bins || is.null(bin_mat)) NA_real_
  else round(mean(bin_mat, na.rm=TRUE), 2)
  mean_r2adj <- if (length(all_r2adj)) round(mean(all_r2adj), 4) else NA_real_

  list(accuracy   = round(correct / n, 4),
       auc        = round(auc, 4),
       avg_bins   = avg_bins,
       mean_R2adj = mean_r2adj,
       disc_time  = round(disc_time, 3),
       clf_time   = round(clf_time, 3))
}

# =============================================================================
# PART 5 — Discretiser registry
# =============================================================================

build_disc_methods <- function(lambda=2) {
  methods <- list(GOLD = function(d,...) gold(d, lambda=lambda))
  if (has_pkg("discretization")) {
    library(discretization, quietly=TRUE)
    methods[["CAIM"]]       <- function(d,...) disc.Topdown(d, method=1)
    methods[["CACC"]]       <- function(d,...) disc.Topdown(d, method=2)
    methods[["Ameva"]]      <- function(d,...) disc.Topdown(d, method=3)
    methods[["ChiMerge"]]   <- function(d,...) chiM(d, alpha=0.05)
    methods[["Chi2"]]       <- function(d,...) chi2(d, delta=0.5, alpha=0.05)
    methods[["ExtendChi2"]] <- function(d,...) extendChi2(d, alp=0.5)
    methods[["MDLP"]]       <- function(d,...) mdlp(d)
  } else {
    message("'discretization' not installed — only GOLD runs in Comparison 1.")
  }
  methods
}

# =============================================================================
# PART 6 — Dataset loader (23 datasets)
#
# Datasets : Iris, BreastCancer, Ionosphere, Glass, Vehicle, Sonar,
#            PimaDiabetes, Vowel, Satellite, Shuttle, BostonHousing, Ozone,
#            Waveform, Normals2D, Circle, Spirals, Cassini, XOR,
#            Ringnorm, Threenorm, Twonorm, Infert, AirQuality
#
# All sourced from mlbench or base R.
# Satellite / Shuttle / BostonHousing / Ozone require mlbench >= 2.1.
#
# Notes on new datasets:
#   Satellite   : 6435 instances, 36 continuous features, 6 land-use classes
#   Shuttle     : 58000 instances, 9 continuous features, 7 classes
#                 (subsampled to 2000 for speed; change n_shuttle below)
#   BostonHousing: 506 instances, 13 continuous features; median house value
#                 binarised at median as "high"/"low" for classification
#   Ozone       : 366 instances, 12 continuous features (after NA removal);
#                 ozone reading binarised at median as "high"/"low"
# =============================================================================

load_datasets <- function(n_shuttle = 2000) {
  ds      <- list()
  ds_meta <- list()   # raw attribute counts before clean()

  # ── helpers ────────────────────────────────────────────────────────────────
  clean <- function(df, class_nm) {
    df <- df[complete.cases(df),, drop = FALSE]
    feat <- setdiff(names(df), class_nm)
    keep <- vapply(feat, function(nm) is.numeric(df[[nm]]), logical(1))
    df   <- df[, c(feat[keep], class_nm), drop = FALSE]
    df[[class_nm]] <- as.factor(df[[class_nm]])
    if (nrow(df) < 50 || ncol(df) < 2) return(NULL)
    df
  }

  safe_mlb <- function(nm, expr) {
    tryCatch({
      need("mlbench"); e <- new.env()
      data(list = nm, package = "mlbench", envir = e)
      expr(e[[nm]])
    }, error = function(e) NULL)
  }

  # record raw attribute counts BEFORE clean(), then clean and store
  add <- function(name, raw_df, class_nm, coerce_cols = FALSE) {
    if (is.null(raw_df)) return()
    df <- raw_df
    if (coerce_cols) {
      for (nm in setdiff(names(df), class_nm))
        df[[nm]] <- suppressWarnings(as.numeric(as.character(df[[nm]])))
    }
    feat     <- setdiff(names(df), class_nm)
    n_attr   <- length(feat)
    n_cont   <- sum(vapply(feat, function(nm) is.numeric(df[[nm]]), logical(1)))
    cleaned  <- clean(df, class_nm)
    if (!is.null(cleaned)) {
      ds[[name]]      <<- cleaned
      ds_meta[[name]] <<- list(n_attr_raw = n_attr, n_cont_raw = n_cont)
    }
  }

  # ── Real datasets ──────────────────────────────────────────────────────────

  add("Iris",        iris,  "Species")

  add("BreastCancer",
      safe_mlb("BreastCancer", function(df) df[, -1]),
      "Class", coerce_cols = TRUE)

  add("Ionosphere",
      safe_mlb("Ionosphere", function(df) df),
      "Class")

  add("Glass",
      safe_mlb("Glass", function(df) df),
      "Type")

  add("Vehicle",
      safe_mlb("Vehicle", function(df) df),
      "Class")

  add("Sonar",
      safe_mlb("Sonar", function(df) df),
      "Class")

  add("PimaDiabetes",
      safe_mlb("PimaIndiansDiabetes", function(df) df),
      "diabetes")

  add("Vowel",
      safe_mlb("Vowel", function(df) df[, -1]),   # drop subject-id column
      "Class")

  # Satellite: 6435 x 36 continuous features, 6 land-use classes
  add("Satellite",
      safe_mlb("Satellite", function(df) df),
      "classes")

  # Shuttle: large dataset — subsample for speed
  add("Shuttle",
      safe_mlb("Shuttle", function(df) {
        set.seed(42)
        df <- df[sample(nrow(df), min(n_shuttle, nrow(df))), ]
        df
      }),
      "Class")

  # BostonHousing: binarise median value at its median
  add("BostonHousing",
      safe_mlb("BostonHousing", function(df) {
        df$medv_class <- as.factor(
          ifelse(df$medv >= median(df$medv), "high", "low"))
        df$medv <- NULL
        df
      }),
      "medv_class")

  # Ozone: binarise daily ozone reading at its median
  add("Ozone",
      safe_mlb("Ozone", function(df) {
        # V4 is the daily ozone reading; keep all numeric columns
        target_col <- "V4"
        if (!target_col %in% names(df)) return(NULL)
        df$ozone_class <- as.factor(
          ifelse(df[[target_col]] >= median(df[[target_col]], na.rm = TRUE),
                 "high", "low"))
        df[[target_col]] <- NULL
        df
      }),
      "ozone_class")

  # ── Synthetic datasets ─────────────────────────────────────────────────────
  synth_add <- function(name, expr) {
    df <- tryCatch({
      need("mlbench"); set.seed(1)
      d <- as.data.frame(expr())
      names(d)[ncol(d)] <- "Class"
      d
    }, error = function(e) NULL)
    add(name, df, "Class")
  }

  synth_add("Waveform",  function() mlbench::mlbench.waveform(300))
  synth_add("Normals2D", function() mlbench::mlbench.2dnormals(300, cl = 4))
  synth_add("Circle",    function() mlbench::mlbench.circle(300, d = 4))
  synth_add("Spirals",   function() mlbench::mlbench.spirals(300, cycles = 1, sd = 0.1))
  synth_add("Cassini",   function() mlbench::mlbench.cassini(300))
  synth_add("XOR",       function() mlbench::mlbench.xor(300, d = 4))
  synth_add("Ringnorm",  function() mlbench::mlbench.ringnorm(300))
  synth_add("Threenorm", function() mlbench::mlbench.threenorm(300))
  synth_add("Twonorm",   function() mlbench::mlbench.twonorm(300))

  # ── Base R datasets ────────────────────────────────────────────────────────
  add("Infert",
      tryCatch({
        df <- infert[, c("age","parity","induced","spontaneous","case")]
        df$case <- as.factor(df$case)
        df
      }, error = function(e) NULL),
      "case")

  add("AirQuality",
      tryCatch({
        df <- na.omit(airquality[, c("Solar.R","Wind","Temp","Ozone")])
        df$OzoneClass <- as.factor(
          ifelse(df$Ozone >= median(df$Ozone), "high", "low"))
        df$Ozone <- NULL
        df
      }, error = function(e) NULL),
      "OzoneClass")

  list(datasets = Filter(Negate(is.null), ds),
       meta     = ds_meta)
}

# =============================================================================
# PART 6b — Table 1 generator (uses raw meta counts)
# =============================================================================

make_table1 <- function(datasets, meta) {
  meta_lookup <- data.frame(
    Dataset = c("Iris","BreastCancer","Ionosphere","Glass","Vehicle","Sonar",
                "PimaDiabetes","Vowel","Satellite","Shuttle","BostonHousing","Ozone",
                "Waveform","Normals2D","Circle","Spirals","Cassini","XOR",
                "Ringnorm","Threenorm","Twonorm","Infert","AirQuality"),
    Source  = c(rep("mlbench", 12), rep("mlbench", 9), rep("base R", 2)),
    Type    = c(rep("Real", 12), rep("Synthetic", 9), rep("Real", 2)),
    stringsAsFactors = FALSE
  )

  rows <- lapply(names(datasets), function(nm) {
    data <- datasets[[nm]];  rc <- meta[[nm]]
    data.frame(
      Dataset      = nm,
      N_Classes    = nlevels(as.factor(data[[ncol(data)]])),
      N_Instances  = nrow(data),
      N_Attributes = rc$n_attr_raw,
      N_Continuous = rc$n_cont_raw,
      stringsAsFactors = FALSE
    )
  })

  t1 <- merge(do.call(rbind, rows), meta_lookup, by = "Dataset", all.x = TRUE)
  t1 <- t1[match(names(datasets), t1$Dataset), ]
  rownames(t1) <- NULL

  cat("\n=== Table 1: Dataset Properties ===\n")
  print(t1, row.names = FALSE)
  write.csv(t1, "table1_datasets.csv", row.names = FALSE)
  invisible(t1)
}

# =============================================================================
# PART 7 — COMPARISON 1: all discretisers, all classifiers
# =============================================================================

run_comparison1 <- function(datasets, disc_methods,
                            clf_types=c("nb","oner","lr","xgb"),
                            k=10, seed=42) {
  clf_available <- clf_types[vapply(clf_types, function(ct)
    switch(ct, nb=TRUE, oner=TRUE,
           lr=has_pkg("nnet"), xgb=has_pkg("xgboost"), FALSE), logical(1))]
  clf_labels <- c(nb="NaiveBayes",lr="LogisticReg",xgb="XGBoost",oner="OneR")
  rows       <- list()
  acc_matrix <- matrix(NA_real_, nrow=length(datasets), ncol=length(disc_methods),
                       dimnames=list(names(datasets), names(disc_methods)))
  auc_matrix <- matrix(NA_real_, nrow=length(datasets), ncol=length(disc_methods),
                       dimnames=list(names(datasets), names(disc_methods)))

  cat("\n",rep("=",65),"\n",sep="")
  cat(" COMPARISON 1 — GOLD vs competing discretisers\n")
  cat(rep("=",65),"\n",sep="")

  for (ds_name in names(datasets)) {
    data <- datasets[[ds_name]]
    cat(sprintf("\n  [%s]  (%d x %d)\n", ds_name, nrow(data), ncol(data)))
    for (disc_nm in names(disc_methods)) {
      cat(sprintf("    %-15s ...", disc_nm))
      res <- tryCatch(
        cv_run(data, disc_fn=disc_methods[[disc_nm]],
               clf_types=clf_available, k=k, seed=seed,
               continuous=FALSE, track_bins=TRUE),
        error=function(e){cat(" ERROR:",conditionMessage(e),"\n");NULL})
      if (is.null(res)) { cat("\n"); next }

      r2_str <- if (!is.na(res$mean_R2adj))
        sprintf("  R2adj=%.3f", res$mean_R2adj) else ""
      cat(sprintf("  bins=%.1f%s  disc=%.2fs",
                  res$avg_bins, r2_str, res$disc_time))

      if ("nb" %in% clf_available) {
        acc_matrix[ds_name, disc_nm] <- res$accuracy["nb"]
        auc_matrix[ds_name, disc_nm] <- res$auc["nb"]
      }

      for (ct in clf_available) {
        cat(sprintf("  %s=%.1f%%/AUC=%.3f",
                    clf_labels[ct], res$accuracy[ct]*100,
                    ifelse(is.na(res$auc[ct]), 0, res$auc[ct])))
        rows[[length(rows)+1]] <- data.frame(
          Dataset      = ds_name,
          Discretizer  = disc_nm,
          Classifier   = clf_labels[ct],
          Accuracy     = res$accuracy[ct],
          AUC          = res$auc[ct],
          Avg_Bins     = res$avg_bins,
          Mean_R2adj   = res$mean_R2adj,
          Disc_Time_s  = res$disc_time,
          Clf_Time_s   = res$clf_time[ct],
          Total_Time_s = round(res$disc_time + res$clf_time[ct], 3),
          stringsAsFactors=FALSE)
      }
      cat("\n")
    }
  }
  list(results=do.call(rbind, rows), acc_matrix=acc_matrix, auc_matrix=auc_matrix)
}

# =============================================================================
# PART 8 — COMPARISON 2: GOLD vs None — accuracy only
# =============================================================================

run_comparison2 <- function(datasets, lambda=2, k=10, seed=42) {
  shared_clf <- c(if(has_pkg("nnet"))    "lr"  else NULL,
                  if(has_pkg("xgboost")) "xgb" else NULL)
  clf_labels <- c(gnb="GaussNB(raw)",lr="LogisticReg",xgb="XGBoost")
  rows <- list()

  cat("\n\n",rep("=",65),"\n",sep="")
  cat(" COMPARISON 2 — GOLD vs None (accuracy + AUC)\n")
  cat(rep("=",65),"\n",sep="")

  for (ds_name in names(datasets)) {
    data <- datasets[[ds_name]]
    cat(sprintf("\n  [%s]\n", ds_name))

    disc_fn <- function(d,...) gold(d, lambda=lambda)
    cat(sprintf("    %-15s ...", "GOLD"))
    res_disc <- tryCatch(
      cv_run(data, disc_fn=disc_fn, clf_types=c("nb",shared_clf),
             k=k, seed=seed, continuous=FALSE, track_bins=FALSE),
      error=function(e){cat(" ERROR:",conditionMessage(e),"\n");NULL})
    if (!is.null(res_disc))
      cat(sprintf("  NB=%.1f%%/AUC=%.3f", res_disc$accuracy["nb"]*100,
                  ifelse(is.na(res_disc$auc["nb"]), 0, res_disc$auc["nb"])),
          if(length(shared_clf))
            paste(sprintf("  %s=%.1f%%/AUC=%.3f", clf_labels[shared_clf],
                          res_disc$accuracy[shared_clf]*100,
                          ifelse(is.na(res_disc$auc[shared_clf]), 0,
                                 res_disc$auc[shared_clf])), collapse=""),
          "\n", sep="")

    cat(sprintf("    %-15s ...", "None (raw)"))
    res_none <- tryCatch(
      cv_run(data, disc_fn=no_disc, clf_types=c("gnb",shared_clf),
             k=k, seed=seed, continuous=TRUE, track_bins=FALSE),
      error=function(e){cat(" ERROR:",conditionMessage(e),"\n");NULL})
    if (!is.null(res_none))
      cat(sprintf("  GNB=%.1f%%/AUC=%.3f", res_none$accuracy["gnb"]*100,
                  ifelse(is.na(res_none$auc["gnb"]), 0, res_none$auc["gnb"])),
          if(length(shared_clf))
            paste(sprintf("  %s=%.1f%%/AUC=%.3f", clf_labels[shared_clf],
                          res_none$accuracy[shared_clf]*100,
                          ifelse(is.na(res_none$auc[shared_clf]), 0,
                                 res_none$auc[shared_clf])), collapse=""),
          "\n", sep="")

    if (is.null(res_disc) || is.null(res_none)) next

    row <- list(Dataset=ds_name)
    # accuracy columns
    row[["Disc_NB"]]      <- res_disc$accuracy["nb"]
    row[["None_GNB"]]     <- res_none$accuracy["gnb"]
    row[["Diff_NB"]]      <- round(res_disc$accuracy["nb"] - res_none$accuracy["gnb"], 4)
    # AUC columns
    row[["AUC_Disc_NB"]]  <- res_disc$auc["nb"]
    row[["AUC_None_GNB"]] <- res_none$auc["gnb"]
    row[["AUC_Diff_NB"]]  <- round(res_disc$auc["nb"] - res_none$auc["gnb"], 4)

    for (ct in shared_clf) {
      lbl <- clf_labels[ct]
      row[[paste0("Disc_", lbl)]]     <- res_disc$accuracy[ct]
      row[[paste0("None_", lbl)]]     <- res_none$accuracy[ct]
      row[[paste0("Diff_", lbl)]]     <- round(res_disc$accuracy[ct] - res_none$accuracy[ct], 4)
      row[[paste0("AUC_Disc_", lbl)]] <- res_disc$auc[ct]
      row[[paste0("AUC_None_", lbl)]] <- res_none$auc[ct]
      row[[paste0("AUC_Diff_", lbl)]] <- round(res_disc$auc[ct] - res_none$auc[ct], 4)
    }
    rows[[length(rows)+1]] <- as.data.frame(row, stringsAsFactors=FALSE)
  }
  do.call(rbind, rows)
}

# =============================================================================
# PART 9 — Statistical tests
# =============================================================================

stat_test_comparison1 <- function(acc_matrix) {
  mat <- acc_matrix[complete.cases(acc_matrix),,drop=FALSE]
  if (nrow(mat) < 5) {cat("\nNot enough datasets.\n");return(invisible(NULL))}

  cat("\n\n",rep("=",65),"\n",sep="")
  cat(sprintf(" Statistical Tests — Comparison 1 (%d datasets)\n", nrow(mat)))
  cat(rep("=",65),"\n",sep="")

  ft <- friedman.test(mat)
  cat(sprintf("\nFriedman test: chi2(%.0f) = %.4f, p = %.4f  %s\n",
              ft$parameter, ft$statistic, ft$p.value,
              ifelse(ft$p.value<0.05,"[SIGNIFICANT]","[not significant]")))

  if (!"GOLD" %in% colnames(mat)) return(invisible(NULL))
  competitors <- setdiff(colnames(mat), "GOLD")
  pvals <- vapply(competitors, function(nm)
    tryCatch(wilcox.test(mat[,"GOLD"], mat[,nm], paired=TRUE, exact=FALSE)$p.value,
             error=function(e) NA_real_), numeric(1))
  adj  <- p.adjust(pvals, method="holm")
  diff <- vapply(competitors, function(nm)
    mean(mat[,"GOLD"] - mat[,nm], na.rm=TRUE), numeric(1))

  pw <- data.frame(
    vs         = competitors,
    Mean_GOLD  = round(mean(mat[,"GOLD"], na.rm=TRUE), 4),
    Mean_Other = round(vapply(competitors, function(nm)
      mean(mat[,nm],na.rm=TRUE), numeric(1)), 4),
    Mean_Diff  = round(diff, 4),
    p_raw      = round(pvals, 4),
    p_Holm     = round(adj,   4),
    Sig        = ifelse(adj<0.05,"YES *","no"),
    stringsAsFactors=FALSE)
  pw <- pw[order(pw$p_Holm),]

  cat("\nPost-hoc Wilcoxon (Holm-corrected): GOLD vs each discretiser\n\n")
  print(pw, row.names=FALSE)
  invisible(list(friedman=ft, pairwise=pw, matrix=mat))
}

stat_test_comparison2 <- function(comp2_results) {
  cat("\n\n",rep("=",65),"\n",sep="")
  cat(" Statistical Tests — Comparison 2 (GOLD vs None)\n\n")

  pairs <- list(
    list(disc="Disc_NB",          none="None_GNB",
         label="NB(disc) vs GaussNB(raw)"),
    list(disc="Disc_LogisticReg", none="None_LogisticReg",
         label="LR(disc) vs LR(raw)"),
    list(disc="Disc_XGBoost",     none="None_XGBoost",
         label="XGB(disc) vs XGB(raw)")
  )
  for (pr in pairs) {
    if (!pr$disc %in% names(comp2_results) ||
        !pr$none %in% names(comp2_results)) next
    d  <- as.numeric(comp2_results[[pr$disc]])
    n  <- as.numeric(comp2_results[[pr$none]])
    ok <- complete.cases(data.frame(d,n))
    if (sum(ok)<5) next
    wt <- tryCatch(wilcox.test(d[ok],n[ok],paired=TRUE,exact=FALSE),
                   error=function(e)NULL)
    if (is.null(wt)) next
    cat(sprintf("  %-35s  mean_diff=%+.4f  p=%.4f  %s\n",
                pr$label, mean(d[ok]-n[ok]), wt$p.value,
                ifelse(wt$p.value<0.05,"[SIGNIFICANT]","[not significant]")))
  }
}

# =============================================================================
# PART 10 — Table generators
# =============================================================================

make_table2 <- function(bench1) {
  df <- bench1$results[bench1$results$Classifier=="NaiveBayes",
                       c("Dataset","Discretizer","Avg_Bins")]
  t2 <- reshape(df, idvar="Dataset", timevar="Discretizer", direction="wide")
  names(t2) <- gsub("Avg_Bins\\.","",names(t2))
  mean_row  <- c("Mean", round(colMeans(t2[,-1],na.rm=TRUE),2))
  t2 <- rbind(t2, setNames(as.list(mean_row), names(t2)))
  cat("\n=== Table 2: Average bins per feature ===\n")
  print(t2, row.names=FALSE);  write.csv(t2,"table2_avg_bins.csv",row.names=FALSE)
  invisible(t2)
}

make_table3 <- function(bench1) {
  # Accuracy table
  t3 <- as.data.frame(round(bench1$acc_matrix, 4))
  t3$Best <- apply(bench1$acc_matrix, 1, function(r) {
    if(all(is.na(r))) return(NA_character_);  names(which.max(r)) })
  mean_vals <- round(colMeans(bench1$acc_matrix,na.rm=TRUE),4)
  t3 <- rbind(t3, c(as.list(mean_vals), Best="—"))
  t3 <- cbind(Dataset=c(rownames(bench1$acc_matrix),"Mean"), t3)
  rownames(t3) <- NULL
  cat("\n=== Table 3: NB accuracy per discretiser ===\n")
  print(t3, row.names=FALSE);  write.csv(t3,"table3_nb_accuracy.csv",row.names=FALSE)

  # AUC table
  if (!is.null(bench1$auc_matrix)) {
    t3b <- as.data.frame(round(bench1$auc_matrix, 4))
    t3b$Best <- apply(bench1$auc_matrix, 1, function(r) {
      if(all(is.na(r))) return(NA_character_);  names(which.max(r)) })
    mean_auc <- round(colMeans(bench1$auc_matrix, na.rm=TRUE), 4)
    t3b <- rbind(t3b, c(as.list(mean_auc), Best="—"))
    t3b <- cbind(Dataset=c(rownames(bench1$auc_matrix),"Mean"), t3b)
    rownames(t3b) <- NULL
    cat("\n=== Table 3b: NB AUC per discretiser ===\n")
    print(t3b, row.names=FALSE);  write.csv(t3b,"table3b_nb_auc.csv",row.names=FALSE)
  }
  invisible(t3)
}

make_table4 <- function(bench2) {
  has_lr  <- "Disc_LogisticReg" %in% names(bench2)
  has_xgb <- "Disc_XGBoost"     %in% names(bench2)
  col_order <- c("Dataset","Disc_NB","None_GNB","Diff_NB")
  if(has_lr)  col_order <- c(col_order,"Disc_LogisticReg","None_LogisticReg","Diff_LR")
  if(has_xgb) col_order <- c(col_order,"Disc_XGBoost","None_XGBoost","Diff_XGB")
  t4 <- bench2[,intersect(col_order,names(bench2)),drop=FALSE]
  num_cols  <- setdiff(names(t4),"Dataset")
  t4[num_cols] <- lapply(t4[num_cols], function(x) round(as.numeric(x),4))
  diff_cols <- grep("^Diff_",names(t4),value=TRUE)
  acc_cols  <- setdiff(num_cols,diff_cols)
  mean_row  <- c(list(Dataset="Mean"),
                 setNames(lapply(acc_cols,  function(c) round(mean(t4[[c]],na.rm=TRUE),4)), acc_cols),
                 setNames(lapply(diff_cols, function(c) round(mean(t4[[c]],na.rm=TRUE),4)), diff_cols))
  t4 <- rbind(t4, as.data.frame(mean_row,stringsAsFactors=FALSE))
  cat("\n=== Table 4: GOLD vs None accuracy ===\n")
  print(t4, row.names=FALSE);  write.csv(t4,"table4_gold_vs_none.csv",row.names=FALSE)
  invisible(t4)
}

make_table5 <- function(bench1) {
  stat1 <- stat_test_comparison1(bench1$acc_matrix)
  t5 <- stat1$pairwise
  names(t5) <- c("vs Method","Mean GOLD","Mean Other","Mean Diff","p (raw)","p (Holm)","Sig.")
  cat("\n=== Table 5: Post-hoc Wilcoxon — Accuracy (GOLD vs each discretiser) ===\n")
  print(t5, row.names=FALSE);  write.csv(t5,"table5_posthoc_acc.csv",row.names=FALSE)

  # repeat for AUC matrix if available
  if (!is.null(bench1$auc_matrix)) {
    stat1_auc <- stat_test_comparison1(bench1$auc_matrix)
    if (!is.null(stat1_auc)) {
      t5b <- stat1_auc$pairwise
      names(t5b) <- c("vs Method","Mean GOLD","Mean Other","Mean Diff","p (raw)","p (Holm)","Sig.")
      cat("\n=== Table 5b: Post-hoc Wilcoxon — AUC (GOLD vs each discretiser) ===\n")
      print(t5b, row.names=FALSE);  write.csv(t5b,"table5_posthoc_auc.csv",row.names=FALSE)
    }
  }
  invisible(stat1)
}

make_table6 <- function(bench2) {
  # helper: build one row of Wilcoxon results
  wilcox_row <- function(d_col, n_col, label, bench2) {
    if (!d_col %in% names(bench2) || !n_col %in% names(bench2)) return(NULL)
    d  <- as.numeric(bench2[[d_col]]);  n <- as.numeric(bench2[[n_col]])
    ok <- complete.cases(data.frame(d,n));  if(sum(ok)<5) return(NULL)
    wt <- tryCatch(wilcox.test(d[ok],n[ok],paired=TRUE,exact=FALSE),
                   error=function(e)NULL)
    if(is.null(wt)) return(NULL)
    data.frame(Pair=label, N=sum(ok),
               Mean_GOLD=round(mean(d[ok]),4), Mean_None=round(mean(n[ok]),4),
               Mean_Diff=round(mean(d[ok]-n[ok]),4),
               W=round(wt$statistic,2), p_value=round(wt$p.value,4),
               Sig=ifelse(wt$p.value<0.05,"YES *","no"),
               stringsAsFactors=FALSE)
  }

  acc_pairs <- list(
    list(d="Disc_NB",          n="None_GNB",          label="NB(disc) vs GaussNB(raw)"),
    list(d="Disc_LogisticReg", n="None_LogisticReg",   label="LR(disc) vs LR(raw)"),
    list(d="Disc_XGBoost",     n="None_XGBoost",       label="XGB(disc) vs XGB(raw)"))

  auc_pairs <- list(
    list(d="AUC_Disc_NB",          n="AUC_None_GNB",          label="NB(disc) vs GaussNB(raw)"),
    list(d="AUC_Disc_LogisticReg", n="AUC_None_LogisticReg",   label="LR(disc) vs LR(raw)"),
    list(d="AUC_Disc_XGBoost",     n="AUC_None_XGBoost",       label="XGB(disc) vs XGB(raw)"))

  # Accuracy Wilcoxon
  t6_acc <- do.call(rbind, Filter(Negate(is.null),
                                  lapply(acc_pairs, function(pr)
                                    wilcox_row(pr$d, pr$n, pr$label, bench2))))
  rownames(t6_acc) <- NULL
  cat("\n=== Table 6: Wilcoxon — Accuracy: GOLD vs None ===\n")
  print(t6_acc, row.names=FALSE)
  write.csv(t6_acc,"table6_wilcoxon_acc.csv",row.names=FALSE)

  # AUC Wilcoxon
  t6_auc <- do.call(rbind, Filter(Negate(is.null),
                                  lapply(auc_pairs, function(pr)
                                    wilcox_row(pr$d, pr$n, pr$label, bench2))))
  if (!is.null(t6_auc)) {
    rownames(t6_auc) <- NULL
    cat("\n=== Table 6b: Wilcoxon — AUC: GOLD vs None ===\n")
    print(t6_auc, row.names=FALSE)
    write.csv(t6_auc,"table6b_wilcoxon_auc.csv",row.names=FALSE)
  }
  invisible(list(acc=t6_acc, auc=t6_auc))
}

# =============================================================================
# RUN
# =============================================================================

cat("Loading datasets...\n")
loaded   <- load_datasets(n_shuttle = 2000)   # increase n_shuttle for final runs
datasets <- loaded$datasets
meta     <- loaded$meta

cat(sprintf("Loaded %d datasets: %s\n\n",
            length(datasets), paste(names(datasets), collapse=", ")))

t1 <- make_table1(datasets, meta)   # Table 1: properties with raw column counts

disc_methods <- build_disc_methods(lambda=0.5)

# Comparison 1
bench1 <- run_comparison1(datasets, disc_methods, k=10, seed=42)
t2 <- make_table2(bench1)
t3 <- make_table3(bench1)
t5 <- make_table5(bench1)

# Comparison 2
bench2 <- run_comparison2(datasets, lambda=2, k=10, seed=42)
t4 <- make_table4(bench2)
t6 <- make_table6(bench2)
stat_test_comparison2(bench2)
