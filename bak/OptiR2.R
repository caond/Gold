# =============================================================================
# OptiR2 — Full Benchmark
#
# COMPARISON 1: OptiR2 vs 7 competing discretisers
#   Classifiers : NaiveBayes (disc), LogisticReg, XGBoost, OneR
#   Metrics     : accuracy, avg bins per feature, wall-clock time
#   Statistics  : Friedman test + Holm-corrected Wilcoxon (OptiR2 vs each)
#
# COMPARISON 2: OptiR2 vs None (no discretisation) — accuracy only
#   Classifiers : GaussianNB (raw), LogisticReg, XGBoost
#   (LR and XGBoost run in both comparisons as shared classifiers)
#   Statistics  : 3 independent Wilcoxon signed-rank tests (no correction)
#
# Datasets : 22 (mlbench + base R)
#
# install.packages(c("discretization", "nnet", "xgboost", "mlbench"))
# =============================================================================

need    <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop(sprintf("Package '%s' not installed. Run: install.packages('%s')", pkg, pkg),
         call. = FALSE)
}
has_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)

# =============================================================================
# PART 1 — OptiR2 discretisation
# =============================================================================

optiR2_discretize_logistic <- function(x, y, min_bin_size = 5, lambda = 1) {
  stopifnot(is.numeric(x), length(x) == length(y))
  ok <- is.finite(x) & !is.na(y);  x <- x[ok];  y <- y[ok]
  if (!length(x)) stop("No complete (x,y) pairs.")
  y <- as.factor(y);  K <- nlevels(y);  N <- length(y)
  penalty_per_bin <- lambda * (K - 1) * log(N) / 2
  ord <- order(x);  xs <- x[ord];  ys <- y[ord]
  if (length(unique(xs)) == 1L)
    return(list(breaks = c(-Inf, Inf), cutpoints = numeric(0)))
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
  interval_ll <- function(j, i) {
    n <- pref_n[i + 1L] - pref_n[j];  if (n < min_bin_size) return(-Inf)
    cnt <- pref_counts[i + 1L, ] - pref_counts[j, ];  pos <- cnt > 0
    sum(cnt[pos] * log(cnt[pos] / n))
  }
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
  if (!is.finite(dp[G + 1L])) stop("DP failed.")
  starts <- integer(0);  i <- G
  while (i > 0L) { j <- last[i + 1L];  starts <- c(j, starts);  i <- j - 1L }
  cutpoints <- if (length(starts) <= 1L) numeric(0) else {
    k <- starts[-1L] - 1L;  (v[k] + v[k + 1L]) / 2
  }
  list(breaks = c(-Inf, cutpoints, Inf), cutpoints = cutpoints)
}

optiR <- function(data, lambda = 1) {
  class_col    <- ncol(data);  min_bin_size <- max(5L, floor(nrow(data) / 20L))
  feature_cols <- seq_len(class_col - 1L)
  cutp         <- vector("list", length(feature_cols))
  Disc.data    <- data;  bin_counts <- integer(0)
  for (i in feature_cols) {
    col <- data[[i]];  if (!is.numeric(col)) next
    res        <- optiR2_discretize_logistic(col, data[[class_col]],
                                             min_bin_size = min_bin_size,
                                             lambda = lambda)
    cutp[[i]]      <- if (length(res$cutpoints) == 0L) "All" else res$cutpoints
    Disc.data[[i]] <- as.integer(cut(col, breaks = res$breaks,
                                     include.lowest = TRUE, right = TRUE))
    bin_counts <- c(bin_counts, length(res$breaks) - 1L)
  }
  list(cutp = cutp, Disc.data = Disc.data,
       avg_bins   = if (length(bin_counts)) round(mean(bin_counts), 2) else NA_real_,
       total_bins = sum(bin_counts))
}

# No-discretisation passthrough
no_disc <- function(data, ...) {
  class_col    <- ncol(data)
  cutp <- lapply(seq_len(class_col - 1L), function(i) "All")
  list(cutp = cutp, Disc.data = data, avg_bins = NA_real_, total_bins = NA_integer_)
}

# =============================================================================
# PART 2 — Classifiers
# =============================================================================

# ---- Discrete Naive Bayes (Laplace) ----------------------------------------
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
  list(log_prior = log_prior, cond = cond, classes = classes,
       class_col = class_col, type = "nb")
}
nb_predict <- function(model, Disc.data) {
  n        <- nrow(Disc.data)
  log_post <- matrix(rep(model$log_prior, each = n), n, length(model$classes),
                     dimnames = list(NULL, model$classes))
  for (i in seq_len(model$class_col - 1L)) {
    ci <- model$cond[[i]];  if (is.null(ci)) next
    col <- Disc.data[[i]]
    for (j in seq_len(n)) {
      b <- as.character(col[j])
      if (b %in% rownames(ci$log_lk))
        log_post[j, ] <- log_post[j, ] + ci$log_lk[b, ]
    }
  }
  model$classes[apply(log_post, 1, which.max)]
}

# ---- Gaussian Naive Bayes (continuous only) --------------------------------
gnb_train <- function(data) {
  class_col <- ncol(data);  y <- as.factor(data[[class_col]])
  classes <- levels(y);  n_cls <- length(classes)
  log_prior <- log(as.numeric(table(y)) / length(y));  names(log_prior) <- classes
  params <- lapply(seq_len(class_col - 1L), function(i) {
    col <- as.numeric(data[[i]])
    setNames(lapply(classes, function(cls) {
      vals <- col[y == cls]
      list(mu = mean(vals, na.rm = TRUE), sigma = max(sd(vals, na.rm = TRUE), 1e-9))
    }), classes)
  })
  list(log_prior = log_prior, params = params, classes = classes,
       class_col = class_col, type = "gnb")
}
gnb_predict <- function(model, data) {
  n        <- nrow(data)
  log_post <- matrix(rep(model$log_prior, each = n), n, length(model$classes),
                     dimnames = list(NULL, model$classes))
  for (i in seq_len(model$class_col - 1L)) {
    col <- as.numeric(data[[i]])
    for (cls in model$classes) {
      p <- model$params[[i]][[cls]]
      log_post[, cls] <- log_post[, cls] +
        dnorm(col, mean = p$mu, sd = p$sigma, log = TRUE)
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
                          data = df, trace = FALSE, maxit = 200)
  list(model = model, class_col = class_col, continuous = continuous, type = "lr")
}
lr_predict <- function(fit, Disc.data) {
  df <- Disc.data
  if (!fit$continuous)
    for (i in seq_len(fit$class_col - 1L)) df[[i]] <- as.factor(df[[i]])
  as.character(predict(fit$model, newdata = df))
}

# ---- XGBoost ---------------------------------------------------------------
xgb_train <- function(Disc.data) {
  need("xgboost");  class_col <- ncol(Disc.data)
  y_raw <- as.factor(Disc.data[[class_col]]);  y_int <- as.integer(y_raw) - 1L
  X     <- as.matrix(Disc.data[, -class_col, drop = FALSE])
  storage.mode(X) <- "double";  n_cls <- nlevels(y_raw)
  params <- Filter(Negate(is.null), list(
    objective   = if (n_cls == 2) "binary:logistic" else "multi:softmax",
    num_class   = if (n_cls > 2) n_cls else NULL,
    eta = 0.1, max_depth = 4, nthread = 1,
    eval_metric = if (n_cls == 2) "error" else "merror"))
  model <- xgboost::xgb.train(params,
                              xgboost::xgb.DMatrix(X, label = y_int),
                              nrounds = 50, verbose = 0)
  list(model = model, classes = levels(y_raw),
       class_col = class_col, n_cls = n_cls, type = "xgb")
}
xgb_predict <- function(fit, Disc.data) {
  X <- as.matrix(Disc.data[, -fit$class_col, drop = FALSE])
  storage.mode(X) <- "double"
  raw <- predict(fit$model, xgboost::xgb.DMatrix(X))
  idx <- if (fit$n_cls == 2) ifelse(raw > 0.5, 2L, 1L) else as.integer(raw) + 1L
  fit$classes[idx]
}

# ---- OneR ------------------------------------------------------------------
oner_train <- function(Disc.data) {
  class_col <- ncol(Disc.data);  y <- as.factor(Disc.data[[class_col]])
  classes <- levels(y);  best_feat <- NULL;  best_rules <- NULL;  best_err <- Inf
  for (i in seq_len(class_col - 1L)) {
    bins  <- Disc.data[[i]];  u <- sort(unique(bins))
    rules <- setNames(vapply(u, function(b) {
      sub <- y[bins == b]
      if (!length(sub)) classes[1L] else names(which.max(table(sub)))
    }, character(1)), as.character(u))
    err <- mean(rules[as.character(bins)] != as.character(y))
    if (err < best_err) { best_err <- err;  best_feat <- i;  best_rules <- rules }
  }
  list(feat = best_feat, rules = best_rules, default = classes[1L],
       class_col = class_col, type = "oner")
}
oner_predict <- function(fit, Disc.data) {
  col <- as.character(Disc.data[[fit$feat]])
  as.character(ifelse(col %in% names(fit$rules), fit$rules[col], fit$default))
}

# ---- dispatch --------------------------------------------------------------
clf_train <- function(type, data, continuous = FALSE) {
  switch(type,
         nb   = nb_train(data),
         gnb  = gnb_train(data),
         lr   = lr_train(data, continuous = continuous),
         xgb  = xgb_train(data),
         oner = oner_train(data),
         stop("Unknown: ", type))
}
clf_predict <- function(fit, data) {
  switch(fit$type,
         nb   = nb_predict(fit, data),
         gnb  = gnb_predict(fit, data),
         lr   = lr_predict(fit, data),
         xgb  = xgb_predict(fit, data),
         oner = oner_predict(fit, data))
}

# =============================================================================
# PART 3 — Cross-validation core
# =============================================================================

apply_breaks <- function(data, cutp, class_col) {
  out <- data
  for (i in seq_len(class_col - 1L)) {
    col <- data[[i]];  if (!is.numeric(col)) next
    cp  <- cutp[[i]]
    br  <- if (identical(cp, "All") || length(cp) == 0) c(-Inf, Inf)
    else c(-Inf, as.numeric(cp), Inf)
    out[[i]] <- as.integer(cut(col, breaks = br,
                               include.lowest = TRUE, right = TRUE))
  }
  out
}

count_bins <- function(cutp) {
  sapply(cutp, function(cp) {
    if (is.null(cp))           return(NA_integer_)
    if (identical(cp, "All"))  return(1L)
    length(cp) + 1L
  })
}

# Generic CV worker.
# clf_types  : classifiers to run
# continuous : if TRUE, skip apply_breaks and pass raw data to classifiers
# track_bins : if TRUE, record avg bins (only meaningful for discretisers)
cv_run <- function(data, disc_fn, clf_types, k = 10, seed = 42,
                   continuous = FALSE, track_bins = TRUE, ...) {
  set.seed(seed)
  n         <- nrow(data);  folds <- sample(rep(seq_len(k), length.out = n))
  class_col <- ncol(data);  y <- as.factor(data[[class_col]])
  correct   <- setNames(integer(length(clf_types)),  clf_types)
  disc_time <- 0
  clf_time  <- setNames(numeric(length(clf_types)),  clf_types)
  all_bins  <- list()

  for (fold in seq_len(k)) {
    train  <- data[folds != fold, , drop = FALSE]
    test   <- data[folds == fold, , drop = FALSE]
    y_test <- as.character(y[folds == fold])

    t0         <- proc.time()["elapsed"]
    disc_train <- tryCatch(disc_fn(train, ...), error = function(e) NULL)
    disc_time  <- disc_time + (proc.time()["elapsed"] - t0)
    if (is.null(disc_train)) next

    if (track_bins) all_bins[[fold]] <- count_bins(disc_train$cutp)

    disc_test <- if (continuous) test else
      apply_breaks(test, disc_train$cutp, class_col)

    for (ct in clf_types) {
      t1  <- proc.time()["elapsed"]
      fit <- tryCatch(clf_train(ct, disc_train$Disc.data, continuous = continuous),
                      error = function(e) NULL)
      if (is.null(fit)) next
      preds <- tryCatch(clf_predict(fit, disc_test), error = function(e) NULL)
      clf_time[ct] <- clf_time[ct] + (proc.time()["elapsed"] - t1)
      if (!is.null(preds))
        correct[ct] <- correct[ct] + sum(preds == y_test)
    }
  }

  bin_mat  <- do.call(rbind, all_bins)
  avg_bins <- if (!track_bins || is.null(bin_mat)) NA_real_
  else round(mean(bin_mat, na.rm = TRUE), 2)

  list(accuracy  = round(correct / n, 4),
       avg_bins  = avg_bins,
       disc_time = round(disc_time, 3),
       clf_time  = round(clf_time, 3))
}

# =============================================================================
# PART 4 — Discretiser registry
# =============================================================================

build_disc_methods <- function(lambda = 1) {
  methods <- list(OptiR2 = function(d, ...) optiR(d, lambda = lambda))
  if (has_pkg("discretization")) {
    library(discretization, quietly = TRUE)
    methods[["CAIM"]]       <- function(d, ...) disc.Topdown(d, method = 1)
    methods[["CACC"]]       <- function(d, ...) disc.Topdown(d, method = 2)
    methods[["Ameva"]]      <- function(d, ...) disc.Topdown(d, method = 3)
    methods[["ChiMerge"]]   <- function(d, ...) chiM(d, alpha = 0.05)
    methods[["Chi2"]]       <- function(d, ...) chi2(d)
    methods[["ExtendChi2"]] <- function(d, ...) extendChi2(d, alp = 0.5)
    methods[["MDLP"]]       <- function(d, ...) mdlp(d)
  } else {
    message("'discretization' not installed — only OptiR2 will run in Comparison 1.")
  }
  methods
}

# =============================================================================
# PART 5 — Dataset loader (22 datasets)
# =============================================================================

load_datasets <- function() {
  ds <- list()
  clean <- function(df, class_nm) {
    df <- df[complete.cases(df), , drop = FALSE]
    feat <- setdiff(names(df), class_nm)
    keep <- vapply(feat, function(nm) is.numeric(df[[nm]]), logical(1))
    df   <- df[, c(feat[keep], class_nm), drop = FALSE]
    df[[class_nm]] <- as.factor(df[[class_nm]])
    if (nrow(df) < 50 || ncol(df) < 2) return(NULL);  df
  }
  safe_mlb <- function(nm, expr) {
    tryCatch({ need("mlbench"); e <- new.env()
    data(list = nm, package = "mlbench", envir = e); expr(e[[nm]])
    }, error = function(e) NULL)
  }
  ds[["Iris"]]        <- clean(iris, "Species")
  ds[["BreastCancer"]]<- safe_mlb("BreastCancer", function(df) {
    df <- df[,-1]
    for (i in seq_len(ncol(df)-1))
      df[[i]] <- suppressWarnings(as.numeric(as.character(df[[i]])))
    clean(df,"Class") })
  #ds[["Ionosphere"]]  <- safe_mlb("Ionosphere",  function(df) clean(df,"Class"))
  #ds[["Glass"]]       <- safe_mlb("Glass",        function(df) clean(df,"Type"))
  #ds[["Vehicle"]]     <- safe_mlb("Vehicle",      function(df) clean(df,"Class"))
  #ds[["Sonar"]]       <- safe_mlb("Sonar",        function(df) clean(df,"Class"))
  #ds[["PimaDiabetes"]]<- safe_mlb("PimaIndiansDiabetes",function(df) clean(df,"diabetes"))
  #ds[["Vowel"]]       <- safe_mlb("Vowel",        function(df) { df<-df[,-1]; clean(df,"Class") })
  ds[["Soybean"]]     <- safe_mlb("Soybean",      function(df) {
    for(i in seq_len(ncol(df)-1))
      df[[i]] <- suppressWarnings(as.numeric(as.character(df[[i]])))
    clean(df,"Class") })
  ds[["Zoo"]]         <- safe_mlb("Zoo",          function(df) {
    df<-df[,-1]
    for(i in seq_len(ncol(df)-1))
      df[[i]] <- suppressWarnings(as.numeric(as.character(df[[i]])))
    clean(df,"type") })
  ds[["HouseVotes"]]  <- safe_mlb("HouseVotes84", function(df) {
    for(i in seq_len(ncol(df)-1)) df[[i]] <- as.numeric(df[[i]])
    clean(df,"Class") })
  ds[["Waveform"]]    <- tryCatch({ need("mlbench"); set.seed(1)
    df <- as.data.frame(mlbench::mlbench.waveform(300))
    names(df)[ncol(df)] <- "Class"; clean(df,"Class") }, error=function(e) NULL)
  ds[["Normals2D"]]   <- tryCatch({ need("mlbench"); set.seed(1)
    df <- as.data.frame(mlbench::mlbench.2dnormals(300,cl=4))
    names(df)[ncol(df)] <- "Class"; clean(df,"Class") }, error=function(e) NULL)
  ds[["Circle"]]      <- tryCatch({ need("mlbench"); set.seed(1)
    df <- as.data.frame(mlbench::mlbench.circle(300,d=4))
    names(df)[ncol(df)] <- "Class"; clean(df,"Class") }, error=function(e) NULL)
  ds[["Spirals"]]     <- tryCatch({ need("mlbench"); set.seed(1)
    df <- as.data.frame(mlbench::mlbench.spirals(300,cycles=1,sd=0.1))
    names(df)[ncol(df)] <- "Class"; clean(df,"Class") }, error=function(e) NULL)
  ds[["Cassini"]]     <- tryCatch({ need("mlbench"); set.seed(1)
    df <- as.data.frame(mlbench::mlbench.cassini(300))
    names(df)[ncol(df)] <- "Class"; clean(df,"Class") }, error=function(e) NULL)
  ds[["XOR"]]         <- tryCatch({ need("mlbench"); set.seed(1)
    df <- as.data.frame(mlbench::mlbench.xor(300,d=4))
    names(df)[ncol(df)] <- "Class"; clean(df,"Class") }, error=function(e) NULL)
  ds[["Ringnorm"]]    <- tryCatch({ need("mlbench"); set.seed(1)
    df <- as.data.frame(mlbench::mlbench.ringnorm(300))
    names(df)[ncol(df)] <- "Class"; clean(df,"Class") }, error=function(e) NULL)
  ds[["Threenorm"]]   <- tryCatch({ need("mlbench"); set.seed(1)
    df <- as.data.frame(mlbench::mlbench.threenorm(300))
    names(df)[ncol(df)] <- "Class"; clean(df,"Class") }, error=function(e) NULL)
  ds[["Twonorm"]]     <- tryCatch({ need("mlbench"); set.seed(1)
    df <- as.data.frame(mlbench::mlbench.twonorm(300))
    names(df)[ncol(df)] <- "Class"; clean(df,"Class") }, error=function(e) NULL)
  ds[["Infert"]]      <- tryCatch({
    df <- infert[,c("age","parity","induced","spontaneous","case")]
    for(i in seq_len(ncol(df)-1)) df[[i]] <- as.numeric(df[[i]])
    df[["case"]] <- as.factor(df[["case"]]); clean(df,"case")
  }, error=function(e) NULL)
  ds[["AirQuality"]]  <- tryCatch({
    df <- na.omit(airquality[,c("Solar.R","Wind","Temp","Ozone")])
    df$OzoneClass <- as.factor(ifelse(df$Ozone>median(df$Ozone),"high","low"))
    df$Ozone <- NULL; clean(df,"OzoneClass")
  }, error=function(e) NULL)
  Filter(Negate(is.null), ds)
}

# =============================================================================
# PART 6 — COMPARISON 1: discretisers vs OptiR2
#   Classifiers : nb, lr, xgb, oner  (all on discretised data)
#   Reports     : accuracy, avg_bins, disc_time, clf_time
# =============================================================================

run_comparison1 <- function(datasets, disc_methods,
                            clf_types = c("nb","oner","lr","xgb"),
                            k = 10, seed = 42) {

  clf_available <- clf_types[vapply(clf_types, function(ct)
    switch(ct, nb=TRUE, oner=TRUE,
           lr=has_pkg("nnet"), xgb=has_pkg("xgboost"), FALSE), logical(1))]

  clf_labels <- c(nb="NaiveBayes", lr="LogisticReg", xgb="XGBoost", oner="OneR")

  rows       <- list()
  # NB accuracy matrix for Friedman / Wilcoxon (rows=datasets, cols=discretisers)
  acc_matrix <- matrix(NA_real_, nrow=length(datasets), ncol=length(disc_methods),
                       dimnames=list(names(datasets), names(disc_methods)))

  cat("\n", rep("=",65), "\n", sep="")
  cat(" COMPARISON 1 — Discretisers vs OptiR2\n")
  cat(rep("=",65), "\n", sep="")

  for (ds_name in names(datasets)) {
    data <- datasets[[ds_name]]
    cat(sprintf("\n  [%s]  (%d x %d)\n", ds_name, nrow(data), ncol(data)))

    for (disc_nm in names(disc_methods)) {
      cat(sprintf("    %-15s ...", disc_nm))
      res <- tryCatch(
        cv_run(data, disc_fn=disc_methods[[disc_nm]],
               clf_types=clf_available, k=k, seed=seed,
               continuous=FALSE, track_bins=TRUE),
        error=function(e) { cat(" ERROR:", conditionMessage(e),"\n"); NULL })
      if (is.null(res)) { cat("\n"); next }

      cat(sprintf("  bins=%.1f  disc=%.2fs", res$avg_bins, res$disc_time))

      # store NB accuracy for statistical tests
      if ("nb" %in% clf_available)
        acc_matrix[ds_name, disc_nm] <- res$accuracy["nb"]

      for (ct in clf_available) {
        cat(sprintf("  %s=%.1f%%", clf_labels[ct], res$accuracy[ct]*100))
        rows[[length(rows)+1]] <- data.frame(
          Dataset      = ds_name,
          Discretizer  = disc_nm,
          Classifier   = clf_labels[ct],
          Accuracy     = res$accuracy[ct],
          Avg_Bins     = res$avg_bins,
          Disc_Time_s  = res$disc_time,
          Clf_Time_s   = res$clf_time[ct],
          Total_Time_s = round(res$disc_time + res$clf_time[ct], 3),
          stringsAsFactors = FALSE)
      }
      cat("\n")
    }
  }

  list(results    = do.call(rbind, rows),
       acc_matrix = acc_matrix)
}

# =============================================================================
# PART 7 — COMPARISON 2: OptiR2 vs None — accuracy only
#   Classifiers : gnb (raw), lr (both), xgb (both)
#   Reports     : accuracy only — no bins, no disc_time
# =============================================================================

run_comparison2 <- function(datasets, lambda = 2,
                            k = 10, seed = 42) {

  # shared classifiers that run on both discretised and raw data
  shared_clf <- c(if(has_pkg("nnet"))    "lr"  else NULL,
                  if(has_pkg("xgboost")) "xgb" else NULL)

  clf_labels <- c(gnb="GaussNB(raw)", lr="LogisticReg", xgb="XGBoost")

  # per-classifier: OptiR2 accuracy and None accuracy across datasets
  # columns: disc_nb, none_gnb, disc_lr, none_lr, disc_xgb, none_xgb
  rows <- list()

  cat("\n\n", rep("=",65), "\n", sep="")
  cat(" COMPARISON 2 — OptiR2 vs None (accuracy only)\n")
  cat(rep("=",65), "\n", sep="")

  for (ds_name in names(datasets)) {
    data <- datasets[[ds_name]]
    cat(sprintf("\n  [%s]\n", ds_name))

    # ---- OptiR2 on discretised data (nb + shared classifiers) --------------
    disc_fn <- function(d, ...) optiR(d, lambda = lambda)
    cat(sprintf("    %-15s ...", "OptiR2"))
    res_disc <- tryCatch(
      cv_run(data, disc_fn=disc_fn,
             clf_types=c("nb", shared_clf),
             k=k, seed=seed, continuous=FALSE, track_bins=FALSE),
      error=function(e) { cat(" ERROR:", conditionMessage(e),"\n"); NULL })
    if (!is.null(res_disc))
      cat(sprintf("  NB=%.1f%%", res_disc$accuracy["nb"]*100),
          if(length(shared_clf))
            paste(sprintf("  %s=%.1f%%", clf_labels[shared_clf],
                          res_disc$accuracy[shared_clf]*100), collapse=""),
          "\n", sep="")

    # ---- None: raw continuous data (gnb + shared classifiers) --------------
    cat(sprintf("    %-15s ...", "None (raw)"))
    res_none <- tryCatch(
      cv_run(data, disc_fn=no_disc,
             clf_types=c("gnb", shared_clf),
             k=k, seed=seed, continuous=TRUE, track_bins=FALSE),
      error=function(e) { cat(" ERROR:", conditionMessage(e),"\n"); NULL })
    if (!is.null(res_none))
      cat(sprintf("  GNB=%.1f%%", res_none$accuracy["gnb"]*100),
          if(length(shared_clf))
            paste(sprintf("  %s=%.1f%%", clf_labels[shared_clf],
                          res_none$accuracy[shared_clf]*100), collapse=""),
          "\n", sep="")

    if (is.null(res_disc) || is.null(res_none)) next

    # store for output table
    row <- list(Dataset = ds_name)
    # NB pair
    row[["Disc_NB"]]  <- res_disc$accuracy["nb"]
    row[["None_GNB"]] <- res_none$accuracy["gnb"]
    row[["Diff_NB"]]  <- round(res_disc$accuracy["nb"] - res_none$accuracy["gnb"], 4)
    # shared classifiers
    for (ct in shared_clf) {
      row[[paste0("Disc_",  clf_labels[ct])]] <- res_disc$accuracy[ct]
      row[[paste0("None_",  clf_labels[ct])]] <- res_none$accuracy[ct]
      row[[paste0("Diff_",  clf_labels[ct])]] <- round(
        res_disc$accuracy[ct] - res_none$accuracy[ct], 4)
    }
    rows[[length(rows)+1]] <- as.data.frame(row, stringsAsFactors=FALSE)
  }

  do.call(rbind, rows)
}

# =============================================================================
# PART 8 — Statistical significance tests
# =============================================================================

# Comparison 1: Friedman + Holm-corrected Wilcoxon (discretisers only)
stat_test_comparison1 <- function(acc_matrix) {
  mat <- acc_matrix[complete.cases(acc_matrix), , drop=FALSE]
  if (nrow(mat) < 5) { cat("\nNot enough datasets for significance tests.\n"); return(invisible(NULL)) }

  cat("\n\n", rep("=",65), "\n", sep="")
  cat(sprintf(" Statistical Tests — Comparison 1 (%d datasets)\n", nrow(mat)))
  cat(rep("=",65), "\n", sep="")

  ft <- friedman.test(mat)
  cat(sprintf("\nFriedman test: chi2(%.0f) = %.4f, p = %.4f  %s\n",
              ft$parameter, ft$statistic, ft$p.value,
              ifelse(ft$p.value < 0.05, "[SIGNIFICANT]", "[not significant]")))

  if (!"OptiR2" %in% colnames(mat)) return(invisible(NULL))
  competitors <- setdiff(colnames(mat), "OptiR2")
  pvals <- vapply(competitors, function(nm)
    tryCatch(wilcox.test(mat[,"OptiR2"], mat[,nm], paired=TRUE, exact=FALSE)$p.value,
             error=function(e) NA_real_), numeric(1))
  adj  <- p.adjust(pvals, method="holm")
  diff <- vapply(competitors, function(nm)
    mean(mat[,"OptiR2"] - mat[,nm], na.rm=TRUE), numeric(1))

  pw <- data.frame(
    vs          = competitors,
    Mean_OptiR2 = round(mean(mat[,"OptiR2"], na.rm=TRUE), 4),
    Mean_Other  = round(vapply(competitors, function(nm)
      mean(mat[,nm], na.rm=TRUE), numeric(1)), 4),
    Mean_Diff   = round(diff, 4),
    p_raw       = round(pvals, 4),
    p_Holm      = round(adj,   4),
    Sig         = ifelse(adj < 0.05, "YES *", "no"),
    stringsAsFactors = FALSE)
  pw <- pw[order(pw$p_Holm), ]

  cat("\nPost-hoc Wilcoxon (Holm-corrected): OptiR2 vs each discretiser\n")
  cat("(positive Mean_Diff = OptiR2 wins)\n\n")
  print(pw, row.names=FALSE)
  invisible(list(friedman=ft, pairwise=pw, matrix=mat))
}

# Comparison 2: 3 independent Wilcoxon tests — no correction needed
stat_test_comparison2 <- function(comp2_results) {
  cat("\n\n", rep("=",65), "\n", sep="")
  cat(" Statistical Tests — Comparison 2 (OptiR2 vs None)\n")
  cat(rep("=",65), "\n\n", sep="")

  pairs <- list(
    list(disc_col="Disc_NB",         none_col="None_GNB",
         label="NB(disc) vs GaussNB(raw)"),
    list(disc_col="Disc_LogisticReg", none_col="None_LogisticReg",
         label="LR(disc) vs LR(raw)"),
    list(disc_col="Disc_XGBoost",     none_col="None_XGBoost",
         label="XGB(disc) vs XGB(raw)")
  )

  for (pr in pairs) {
    if (!pr$disc_col %in% names(comp2_results) ||
        !pr$none_col %in% names(comp2_results)) next
    d <- comp2_results[[pr$disc_col]]
    n <- comp2_results[[pr$none_col]]
    ok <- complete.cases(data.frame(d, n))
    if (sum(ok) < 5) next
    wt <- tryCatch(wilcox.test(d[ok], n[ok], paired=TRUE, exact=FALSE),
                   error=function(e) NULL)
    if (is.null(wt)) next
    cat(sprintf("  %-35s  mean_diff=%+.4f  p=%.4f  %s\n",
                pr$label,
                mean(d[ok] - n[ok]),
                wt$p.value,
                ifelse(wt$p.value < 0.05, "[SIGNIFICANT]", "[not significant]")))
  }
}

# =============================================================================
# PART 9 — Summary printers
# =============================================================================

print_summary1 <- function(results, acc_matrix) {
  cat("\n\n", rep("=",65), "\n", sep="")
  cat(" Comparison 1 Summary — mean NB accuracy per discretiser\n")
  cat(rep("=",65), "\n\n", sep="")
  means     <- sort(colMeans(acc_matrix, na.rm=TRUE), decreasing=TRUE)
  mean_bins <- vapply(names(means), function(disc) {
    b <- results$Avg_Bins[results$Discretizer==disc & results$Classifier=="NaiveBayes"]
    if(length(b)) round(mean(b,na.rm=TRUE),2) else NA }, numeric(1))
  wins <- vapply(names(means), function(disc) {
    sum(apply(acc_matrix, 1, function(r) {
      if(all(is.na(r))) return(FALSE)
      !is.na(r[disc]) && r[disc]==max(r,na.rm=TRUE)
    }))
  }, integer(1))
  print(data.frame(Rank=seq_along(means), Discretizer=names(means),
                   Mean_NB_Acc=round(means,4), Dataset_Wins=wins,
                   Avg_Bins=mean_bins, stringsAsFactors=FALSE), row.names=FALSE)
}

print_summary2 <- function(comp2_results) {
  cat("\n\n", rep("=",65), "\n", sep="")
  cat(" Comparison 2 Summary — OptiR2 vs None (mean accuracy diff)\n")
  cat(rep("=",65), "\n\n", sep="")

  diff_cols <- grep("^Diff_", names(comp2_results), value=TRUE)
  summary_rows <- lapply(diff_cols, function(dc) {
    lbl <- sub("^Diff_","",dc)
    data.frame(Classifier_Pair=lbl,
               Mean_Diff=round(mean(comp2_results[[dc]],na.rm=TRUE),4),
               Pct_OptiR2_Wins=round(
                 mean(comp2_results[[dc]] > 0, na.rm=TRUE)*100, 1),
               stringsAsFactors=FALSE)
  })
  df <- do.call(rbind, summary_rows)
  df$Note <- ifelse(df$Mean_Diff > 0, "disc wins", "raw wins")
  print(df, row.names=FALSE)
  cat("\n  (positive Mean_Diff = discretised OptiR2 outperforms raw data)\n")
}

# =============================================================================
# RUN
# =============================================================================

cat("Loading datasets...\n")
datasets <- load_datasets()
datasets<-datasets[1:5]

cat(sprintf("Loaded %d datasets: %s\n",
            length(datasets), paste(names(datasets), collapse=", ")))

disc_methods <- build_disc_methods(lambda=1)

# ── Comparison 1: all discretisers ──────────────────────────────────────────
bench1 <- run_comparison1(datasets, disc_methods, k=10, seed=42)
print_summary1(bench1$results, bench1$acc_matrix)
stat_test_comparison1(bench1$acc_matrix)

# ── Comparison 2: OptiR2 vs None (accuracy only) ────────────────────────────
bench2 <- run_comparison2(datasets, lambda=1, k=10, seed=42)
print_summary2(bench2)
stat_test_comparison2(bench2)


#build Table 2

# ── Generate Table 2: avg bins per feature ──────────────────────────────────
# Filter to NaiveBayes rows only (avg_bins is the same regardless of classifier)
bins_df <- bench1$results[bench1$results$Classifier == "NaiveBayes",c("Dataset", "Discretizer", "Avg_Bins")]

# Reshape long -> wide: rows = datasets, columns = discretisers
table2 <- reshape(bins_df,
                  idvar   = "Dataset",
                  timevar = "Discretizer",
                  direction = "wide")

# Clean up column names
names(table2) <- gsub("Avg_Bins\\.", "", names(table2))
# Add a mean row at the bottom
mean_row        <- c("Mean", round(colMeans(table2[,-1], na.rm=TRUE), 2))
table2          <- rbind(table2, setNames(as.list(mean_row), names(table2)))
# Print
cat("\n=== Table 2: Average bins per feature (10-fold CV) ===\n")
print(table2, row.names=FALSE)


# ── Generate Table 3: NB accuracy per discretiser per dataset ───────────────

# acc_matrix is already rows=datasets, cols=discretisers
table3 <- as.data.frame(bench1$acc_matrix)

# Round all values
table3 <- round(table3, 4)

# Add a "Best" column showing which discretiser won each dataset
table3$Best <- apply(bench1$acc_matrix, 1, function(r) {
  if (all(is.na(r))) return(NA_character_)
  names(which.max(r))
})

# Add a mean row at the bottom
mean_vals       <- round(colMeans(bench1$acc_matrix, na.rm = TRUE), 4)
mean_row        <- c(as.list(mean_vals), Best = "—")
table3          <- rbind(table3, setNames(mean_row, names(table3)))
rownames(table3)[nrow(table3)] <- "Mean"

# Add dataset names as a proper column
table3 <- cbind(Dataset = c(rownames(bench1$acc_matrix), "Mean"), table3)
rownames(table3) <- NULL

# Print
cat("\n=== Table 3: 10-fold CV Naive Bayes accuracy (bold = best per dataset) ===\n")
print(table3, row.names = FALSE)
# Count wins per discretiser (excluding the Mean row)
wins <- table(table3$Best[table3$Dataset != "Mean"])
cat("\n=== Dataset wins per discretiser ===\n")
print(sort(wins, decreasing = TRUE))

# Export to CSV
write.csv(table3, "table3_nb_accuracy.csv", row.names = FALSE)



