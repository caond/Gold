# GOLD core algorithm correctness tests.  Run after source('GOLD.R').
gold_self_test <- function() {
  # Closed-form LOO Brier equals observation-by-observation calculation.
  set.seed(2027)
  for (K in 2:5) for (n in 1:25) {
    y <- sample.int(K, n, replace=TRUE)
    ct <- tabulate(y, nbins=K)
    d <- n - 1 + K*0.5
    direct <- 0
    for (c in y) {
      q <- (ct + 0.5 - as.integer(seq_len(K)==c))/d
      direct <- direct + sum((q - as.integer(seq_len(K)==c))^2)
    }
    stopifnot(isTRUE(all.equal(direct, gold_bin_loss(ct), tolerance=1e-10)))
  }
  # Exhaustive enumeration validates exact optimality for small tied data.
  set.seed(2028)
  for (trial in seq_len(30L)) {
    G <- sample(3:8,1)
    counts <- sample(2:5,G,replace=TRUE)
    x <- rep(seq_len(G),counts)
    y <- factor(sample(c('A','B','C'),length(x),replace=TRUE))
    nmin <- 2L
    fit <- gold_discretize(x,y,min_bin_size=nmin)
    K <- nlevels(y)
    all_values <- unique(x)
    best <- Inf
    for (mask in 0:(2^(G-1)-1)) {
      split_after <- which(as.integer(intToBits(mask)[seq_len(G-1)]) == 1L)
      boundary <- c(0L,split_after,G)
      if (any(diff(boundary) < 1L)) next
      start <- boundary[-length(boundary)] + 1L
      end <- boundary[-1L]
      binloss <- 0
      feasible <- TRUE
      for (z in seq_along(start)) {
        sel <- x >= all_values[start[z]] & x <= all_values[end[z]]
        if (sum(sel) < nmin) {feasible <- FALSE; break}
        binloss <- binloss + gold_bin_loss(tabulate(as.integer(y[sel]),nbins=K))
      }
      if (feasible) best <- min(best,binloss+length(split_after)*log(length(x))/2)
    }
    stopifnot(isTRUE(all.equal(-fit$objective,best,tolerance=1e-8)))
    tr <- data.frame(x=x, Class=y)
    fitted <- gold(tr)
    mapped <- gold_transform(tr, fitted$cutp)
    stopifnot(identical(as.integer(mapped$x),as.integer(fitted$Disc.data$x)))
  }
  TRUE
}
