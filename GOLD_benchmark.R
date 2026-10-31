# GOLD benchmark: reproduction of the classification experiments.
# Requires GOLD.R and GOLD_tests.R in the working directory.
# Set GOLD_RUN_BENCHMARK <- TRUE at the bottom to run experiments.
# The algorithm itself is implemented in GOLD.R.
source('GOLD.R')

need <- function(pkg) {
  if (!requireNamespace(pkg, quietly=TRUE))
    stop(sprintf("Package '%s' is required",pkg),call.=FALSE)
}
has_pkg <- function(pkg) requireNamespace(pkg, quietly=TRUE)

# =============================================================================
# Classifiers and evaluation
# =============================================================================

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


# ---- Logistic Regression ---------------------------------------------------
# Discretised logistic regression: every interval is a FACTOR (one-hot contrasts).
# Raw logistic regression (Comparison 2): original features remain numeric.
lr_train <- function(Disc.data, continuous = FALSE) {
  need("nnet")
  class_col <- ncol(Disc.data)
  df <- as.data.frame(Disc.data)
  df[[class_col]] <- droplevels(as.factor(df[[class_col]]))
  feat_cols <- seq_len(class_col-1L)
  bin_levels <- vector("list", length(feat_cols))
  if (!continuous) {
    for (j in feat_cols) {
      bin_levels[[j]] <- sort(unique(as.character(df[[j]])))
      df[[j]] <- factor(as.character(df[[j]]), levels=bin_levels[[j]])
    }
  }
  # A constant feature has no estimable contrast; drop from model formula.
  active <- if (continuous) {
    feat_cols[vapply(df[feat_cols], function(z) length(unique(z)) > 1L, logical(1))]
  } else {
    feat_cols[lengths(bin_levels) > 1L]
  }
  response_name <- names(df)[class_col]
  formula <- if (length(active)) {
    stats::reformulate(names(df)[active], response=response_name)
  } else {
    stats::as.formula(paste0("`", response_name, "` ~ 1"))
  }

  mm <- model.matrix(formula, data = df)
  K <- nlevels(df[[class_col]])
  n_parameters <- ncol(mm) * K
  model <- nnet::multinom(
    formula,
    data = df,
    trace = FALSE,
    maxit = 200,
    MaxNWts = max(1000L, 2L * n_parameters)
  )
  
  list(model=model, class_col=class_col, classes=levels(df[[class_col]]),
       active=active, bin_levels=bin_levels, continuous=continuous, type="lr")
}
lr_newdata <- function(fit, data) {
  df <- as.data.frame(data)
  if (!fit$continuous) for (j in seq_len(fit$class_col-1L)) {
    df[[j]] <- factor(as.character(df[[j]]),levels=fit$bin_levels[[j]])
    if (anyNA(df[[j]])) stop("Unseen logistic-regression bin code")
  }
  df
}
lr_predict <- function(fit, Disc.data) {
  as.character(predict(fit$model, newdata=lr_newdata(fit,Disc.data)))
}
lr_predict_prob <- function(fit, Disc.data) {
  raw <- predict(fit$model, newdata=lr_newdata(fit,Disc.data),type="probs")
  if (length(fit$classes)==2L) {
    p <- as.numeric(raw)
    ans <- cbind(1-p,p)
  } else {
    ans <- as.matrix(raw)
    if (is.null(dim(raw)) && nrow(Disc.data)==1L)
      ans <- matrix(as.numeric(raw),nrow=1L)
  }
  if(nrow(ans)!=nrow(Disc.data) || ncol(ans)!=length(fit$classes))
    stop("Logistic regression returned incorrectly shaped probabilities")
  colnames(ans) <- fit$classes
  ans
}
# ---- XGBoost ---------------------------------------------------------------
xgb_train <- function(Disc.data) {
  need("xgboost");  class_col <- ncol(Disc.data)
  y_raw <- as.factor(Disc.data[[class_col]]);  y_int <- as.integer(y_raw) - 1L
  X <- as.matrix(Disc.data[,-class_col,drop=FALSE])
  storage.mode(X) <- "double";  n_cls <- nlevels(y_raw)
  params <- Filter(Negate(is.null), list(
    objective   = if (n_cls==2) "binary:logistic" else "multi:softprob",
    num_class   = if (n_cls>2) n_cls else NULL,
    eta=0.1, max_depth=4, nthread=1,
    eval_metric = if (n_cls==2) "error" else "merror"))
  model <- xgboost::xgb.train(params,
                              xgboost::xgb.DMatrix(X, label=y_int), nrounds=50, verbose=0)
  list(model=model, classes=levels(y_raw),
       class_col=class_col, n_cls=n_cls, type="xgb")
}
xgb_predict_prob <- function(fit, Disc.data) {
  X <- as.matrix(Disc.data[, -fit$class_col, drop=FALSE])
  storage.mode(X) <- "double"
  raw <- predict(fit$model, xgboost::xgb.DMatrix(X))
  if (fit$n_cls == 2L) {
    p <- as.numeric(raw)
    prob <- cbind(1-p, p)
  } else if (is.matrix(raw)) {
    prob <- raw
  } else {
    prob <- matrix(raw, nrow=nrow(X), ncol=fit$n_cls, byrow=TRUE)
  }
  stopifnot(nrow(prob)==nrow(X), ncol(prob)==fit$n_cls)
  colnames(prob) <- fit$classes
  prob
}
xgb_predict <- function(fit, Disc.data) {
  prob <- xgb_predict_prob(fit, Disc.data)
  fit$classes[max.col(prob, ties.method="first")]
}

# ---- OneR ------------------------------------------------------------------
oner_train <- function(Disc.data) {
  class_col <- ncol(Disc.data)
  y <- droplevels(as.factor(Disc.data[[class_col]]))
  classes <- levels(y)
  best_feat <- NA_integer_; best_rules <- NULL; best_err <- Inf
  best_probs <- NULL
  for (i in seq_len(class_col-1L)) {
    bins <- as.character(Disc.data[[i]])
    u <- unique(bins)
    tb <- table(factor(bins,levels=u), y)
    rules <- setNames(classes[max.col(tb, ties.method="first")],u)
    err <- mean(unname(rules[bins]) != as.character(y))
    if (err < best_err) {
      best_err <- err; best_feat <- i; best_rules <- rules
      # Laplace-smoothed empirical class probabilities for AUC
      best_probs <- (tb+1)/(rowSums(tb)+length(classes))
    }
  }
  list(feat=best_feat, rules=best_rules, probs=best_probs,
       classes=classes, prior=(as.numeric(table(y))+1)/(length(y)+length(classes)),
       default=classes[which.max(table(y))], class_col=class_col, type="oner")
}
oner_predict <- function(fit, Disc.data) {
  bins <- as.character(Disc.data[[fit$feat]])
  p <- unname(fit$rules[bins])
  p[is.na(p)] <- fit$default
  p
}
oner_predict_prob <- function(fit, Disc.data) {
  bins <- as.character(Disc.data[[fit$feat]])
  prob <- matrix(rep(fit$prior, each=length(bins)), nrow=length(bins),
                 dimnames=list(NULL, fit$classes))
  idx <- match(bins, rownames(fit$probs))
  ok <- which(!is.na(idx))
  if(length(ok)) prob[ok,] <- fit$probs[idx[ok],,drop=FALSE]
  prob
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
  if(is.null(prob_mat) || !has_pkg("pROC")) return(NA_real_)
  prob_mat <- as.matrix(prob_mat)
  labs <- as.character(labels)
  classes <- colnames(prob_mat)
  if(is.null(classes) || ncol(prob_mat)<2L ||
     anyNA(prob_mat) || any(!is.finite(prob_mat)) ||
     !all(unique(labs) %in% classes) || length(unique(labs))<2L) return(NA_real_)
  if(ncol(prob_mat)==2L) {
    # Fixed orientation: class 2 is positive, its probability is the score.
    return(tryCatch(as.numeric(pROC::auc(pROC::roc(
      response=factor(labs, levels=classes), predictor=prob_mat[,2],
      levels=classes, direction="<", quiet=TRUE))),error=function(e) NA_real_))
  }
  if(length(unique(labs))!=length(classes)) return(NA_real_)
  tryCatch(as.numeric(pROC::multiclass.roc(
    response=factor(labs,levels=classes), predictor=prob_mat[,classes,drop=FALSE])$auc),
    error=function(e) NA_real_)
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

cv_run <- function(data, disc_fn, clf_types, k=5, seed=42,
                   continuous=FALSE, track_bins=TRUE, ...) {
  n <- nrow(data); class_col <- ncol(data)
  y <- droplevels(as.factor(data[[class_col]])); classes <- levels(y)
  if (min(table(y)) < k)
    stop("At least one class has fewer than ",k," observations; cannot form stratified ",k,"-fold CV")
  set.seed(seed)
  folds <- integer(n)
  for (idx in split(seq_len(n),y))
    folds[idx] <- sample(rep(seq_len(k),length.out=length(idx)))
  # Store results by original observation ID; guarantees full OOF coverage.
  predictions <- setNames(lapply(clf_types,function(ct) rep(NA_character_,n)),clf_types)
  probabilities <- setNames(lapply(clf_types,function(ct) {
    matrix(NA_real_,nrow=n,ncol=length(classes),dimnames=list(NULL,classes))
  }),clf_types)
  disc_time <- 0; bin_counts <- numeric(0); pbs_scores <- numeric(0)
  for (fold in seq_len(k)) {
    tr <- which(folds!=fold); te <- which(folds==fold)
    train <- data[tr,,drop=FALSE]; test <- data[te,,drop=FALSE]
    t0 <- proc.time()[["elapsed"]]
    fitted_disc <- tryCatch(disc_fn(train,...), error=function(e)
      stop("Discretiser failed in fold ",fold,": ",conditionMessage(e)))
    disc_time <- disc_time + proc.time()[["elapsed"]]-t0
    if (is.null(fitted_disc$Disc.data) || nrow(fitted_disc$Disc.data)!=length(tr))
      stop("Discretiser returned invalid training data in fold ",fold)
    if (!continuous) {
      # Force the same stored boundaries on train and test.
      reconstructed <- apply_breaks(train, fitted_disc$cutp, class_col)
      for (j in seq_len(class_col-1L)) if (is.numeric(train[[j]]) &&
                                           !identical(as.integer(reconstructed[[j]]),as.integer(fitted_disc$Disc.data[[j]])))
        stop("Training/test bin-label mismatch in fold ",fold,", feature ",j)
      test_disc <- apply_breaks(test,fitted_disc$cutp,class_col)
    } else test_disc <- test
    if (track_bins) {
      bin_counts <- c(bin_counts, count_bins(fitted_disc$cutp))
      if (!is.null(fitted_disc$mean_loo_brier) &&
          length(fitted_disc$mean_loo_brier)==1L && is.finite(fitted_disc$mean_loo_brier))
        pbs_scores <- c(pbs_scores,fitted_disc$mean_loo_brier)
    }
    for (ct in clf_types) {
      fit <- tryCatch(clf_train(ct,fitted_disc$Disc.data,continuous=continuous),
                      error=function(e) stop(ct," training failed, fold ",fold,": ",conditionMessage(e)))
      pred <- tryCatch(clf_predict(fit,test_disc),
                       error=function(e) stop(ct," prediction failed, fold ",fold,": ",conditionMessage(e)))
      pr <- tryCatch(clf_predict_prob(fit,test_disc),
                     error=function(e) stop(ct," probability failed, fold ",fold,": ",conditionMessage(e)))
      if(length(pred)!=length(te) || anyNA(pred) || is.null(pr))
        stop(ct," returned missing or incomplete predictions in fold ",fold)
      pr <- as.matrix(pr)
      if(nrow(pr)!=length(te) || is.null(colnames(pr)) ||
         !setequal(colnames(pr),classes) || any(!is.finite(pr)))
        stop(ct," returned invalid probability matrix in fold ",fold)
      pr <- pr[,classes,drop=FALSE]
      if(any(pr < -1e-8) || any(pr > 1+1e-8) || any(abs(rowSums(pr)-1)>1e-5))
        stop(ct," returned invalid class probabilities in fold ",fold)
      predictions[[ct]][te] <- as.character(pred)
      probabilities[[ct]][te,] <- pr
    }
  }
  acc <- setNames(vapply(clf_types,function(ct) {
    if(anyNA(predictions[[ct]])) stop("Incomplete OOF predictions for ",ct)
    mean(predictions[[ct]]==as.character(y))
  }, numeric(1)),clf_types)
  auc <- setNames(vapply(clf_types,function(ct)
    compute_auc(probabilities[[ct]],as.character(y)),numeric(1)),clf_types)
  list(accuracy=round(acc,4),auc=round(auc,4),
       avg_bins=if(length(bin_counts)) round(mean(bin_counts,na.rm=TRUE),2) else NA_real_,
       mean_R2adj=NA_real_,
       mean_loo_brier=if(length(pbs_scores)) round(mean(pbs_scores),6) else NA_real_,
       disc_time=round(disc_time,3))
}

# The discretization::disc.Topdown() cutp vector contains the MINIMUM and
# MAXIMUM observed feature values as well as interior cutpoints.  Our CV
# apply_breaks() expects INTERIOR cutpoints only.  Strip the endpoints and
# reconstruct training bins using the SAME transformation as test bins.
# This prevents train/test bin-label shifts that can reverse predictions.
topdown_cv_safe <- function(data, method=3L) {
  need("discretization")
  fit <- discretization::disc.Topdown(data, method=method)
  p <- ncol(data) - 1L
  cuts <- vector("list", p)
  for (j in seq_len(p)) {
    if (!is.numeric(data[[j]])) {
      cuts[[j]] <- "All"
      next
    }
    cp <- fit$cutp[[j]]
    if (is.null(cp) || !length(cp)) {
      cuts[[j]] <- "All"
      next
    }
    cp <- suppressWarnings(as.numeric(cp))
    cp <- cp[is.finite(cp)]
    # The method documents these as: min, interior cuts, max.
    cp <- sort(unique(cp))
    internal <- if (length(cp) > 2L) cp[-c(1L, length(cp))] else numeric(0)
    cuts[[j]] <- if(length(internal)) internal else "All"
  }
  # Exactly the same bin encoding is used for training and test folds.
  train_disc <- apply_breaks(data, cuts, ncol(data))
  list(cutp=cuts, Disc.data=train_disc)
}

# =============================================================================
# PART 5 — Discretiser registry
# =============================================================================


# ---------------------------------------------------------------------------
# CV-safe bridge for iterative Chi2 / Extended Chi2.
# Their $cutp may refer to INTERMEDIATE discretised bin codes rather than the
# original numeric scale. Recover original-scale cutpoints from the final
# training assignments and verify exact reconstruction before CV.
# ---------------------------------------------------------------------------
chi_cv_safe <- function(data, method = c("Chi2", "ExtendChi2")) {
  method <- match.arg(method)
  need("discretization")
  fit <- if (method == "Chi2") {
    discretization::chi2(data, alp = 0.5, del = 0.05)
  } else {
    discretization::extendChi2(data, alp = 0.5)
  }
  p <- ncol(data) - 1L
  orig <- as.data.frame(data)
  assigned <- as.data.frame(fit$Disc.data)
  cuts <- vector("list", p)
  for (i in seq_len(p)) {
    if (!is.numeric(orig[[i]])) next
    x <- orig[[i]]
    z <- assigned[[i]]
    if (length(z) != length(x) || anyNA(x) || anyNA(z)) {
      stop(method, " produced missing or inconsistent training bins for feature ", i)
    }
    ord <- order(x)
    xs <- x[ord]
    zs <- as.character(z[ord])
    # A cutpoint must not split identical original feature values.
    if (any(xs[-1L] == xs[-length(xs)] & zs[-1L] != zs[-length(zs)])) {
      stop(method, " assigned different bins to identical feature values: ", i)
    }
    transition <- which(zs[-1L] != zs[-length(zs)])
    # One interval may not occur in multiple disconnected ranges.
    runs <- rle(zs)$values
    if (anyDuplicated(runs)) {
      stop(method, " produced noncontiguous interval labels: ", i)
    }
    cuts[[i]] <- if (length(transition)) {
      xs[transition] + (xs[transition + 1L] - xs[transition]) / 2
    } else "All"
  }
  disc_train <- apply_breaks(orig, cuts, ncol(orig))
  # Compare partitions via pairwise label equivalence, allowing arbitrary
  # renumbering of the original method's interval identifiers.
  for (i in seq_len(p)) {
    if (!is.numeric(orig[[i]])) next
    old <- as.character(assigned[[i]])
    new <- as.character(disc_train[[i]])
    if (length(unique(old)) != length(unique(new)) ||
        any(vapply(split(old, new), function(z) length(unique(z)) != 1L,
                   logical(1)))) {
      stop(method, " failed train-bin reconstruction for feature ", i)
    }
  }
  list(cutp = cuts, Disc.data = disc_train)
}

build_disc_methods <- function() {
  methods <- list(GOLD = function(d,...) gold(d))
  if (has_pkg("discretization")) {
    library(discretization, quietly=TRUE)
    methods[["CAIM"]]       <- function(d,...) topdown_cv_safe(d, method=1)
    #methods[["CACC"]]       <- function(d, ...)  topdown_cv_safe(d, method=2)
    methods[["Ameva"]]      <- function(d,...) topdown_cv_safe(d, method=3)
    methods[["ChiMerge"]]   <- function(d,...) chiM(d, alpha=0.05)
    methods[["Chi2"]]       <- function(d, ...) chi_cv_safe(d, "Chi2")
    methods[["ExtendChi2"]] <- function(d, ...) chi_cv_safe(d, "ExtendChi2")
    methods[["MDLP"]]       <- function(d,...) mdlp(d)
  } else {
    message("'discretization' not installed — only GOLD runs in Comparison 1.")
  }
  methods
}

# =============================================================================
# PART 6 — Dataset loader (20 datasets)
#
# Datasets : Iris, BreastCancer, Ionosphere, Glass, Vehicle,
#            PimaDiabetes, Vowel, Satellite, Shuttle, BostonHousing, Ozone,
#            Waveform, Normals2D, Circle, Spirals, Cassini,
#            Ringnorm, Threenorm, Twonorm, AirQuality
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
        #df
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
  synth_add("Ringnorm",  function() mlbench::mlbench.ringnorm(300))
  synth_add("Threenorm", function() mlbench::mlbench.threenorm(300))
  synth_add("Twonorm",   function() mlbench::mlbench.twonorm(300))
  
  # ── Base R datasets ────────────────────────────────────────────────────────
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
                            clf_types = c("nb", "oner", "lr", "xgb"),
                            k = 10, seed = 42) {
  
  clf_available <- clf_types[vapply(clf_types, function(ct)
    switch(ct, nb = TRUE, oner = TRUE,
           lr  = has_pkg("nnet"), xgb = has_pkg("xgboost"), FALSE), logical(1))]
  
  clf_labels <- c(nb = "NaiveBayes", lr = "LogisticRegFactor",
                  xgb = "XGBoost",   oner = "OneR")
  
  rows <- list()
  disc_nms <- names(disc_methods)
  ds_nms   <- names(datasets)
  
  # ── One acc/auc matrix per classifier ─────────────────────────────────────
  make_mat <- function() {
    matrix(NA_real_,
           nrow = length(ds_nms),
           ncol = length(disc_nms),
           dimnames = list(ds_nms, disc_nms))
  }
  acc_matrices <- setNames(lapply(clf_available, function(ct) make_mat()), clf_available)
  auc_matrices <- setNames(lapply(clf_available, function(ct) make_mat()), clf_available)
  
  cat("\n", rep("=", 65), "\n", sep = "")
  cat(" COMPARISON 1 — GOLD vs competing discretisers\n")
  cat(rep("=", 65), "\n", sep = "")
  
  for (ds_name in ds_nms) {
    data <- datasets[[ds_name]]
    cat(sprintf("\n  [%s]  (%d x %d)\n", ds_name, nrow(data), ncol(data)))
    
    for (disc_nm in disc_nms) {
      cat(sprintf("    %-15s ...", disc_nm))
      res <- tryCatch(
        cv_run(data, disc_fn = disc_methods[[disc_nm]],
               clf_types = clf_available, k = k, seed = seed,
               continuous = FALSE, track_bins = TRUE),
        error = function(e) { cat(" ERROR:", conditionMessage(e), "\n"); NULL })
      if (is.null(res)) { cat("\n"); next }
      stopifnot(length(res$mean_loo_brier) == 1L)
      
      r2_str <- if (!is.na(res$mean_R2adj))
        sprintf("  R2adj=%.3f", res$mean_R2adj) else ""
      cat(sprintf("  bins=%.1f%s  disc=%.2fs",
                  res$avg_bins, r2_str, res$disc_time))
      
      for (ct in clf_available) {
        acc_matrices[[ct]][ds_name, disc_nm] <- res$accuracy[ct]
        auc_matrices[[ct]][ds_name, disc_nm] <- res$auc[ct]
        
        auc_str <- if (is.na(res$auc[ct])) "NA"
        else sprintf("%.3f", res$auc[ct])
        cat(sprintf("  %s=%.1f%%/AUC=%s",
                    clf_labels[ct], res$accuracy[ct] * 100, auc_str))
        
        rows[[length(rows) + 1]] <- data.frame(
          Dataset     = ds_name,
          Discretizer = disc_nm,
          Classifier  = clf_labels[ct],
          Accuracy    = res$accuracy[ct],
          AUC         = res$auc[ct],
          Avg_Bins    = res$avg_bins,
          Mean_R2adj  = res$mean_R2adj,
          Mean_LOO_Brier = res$mean_loo_brier,
          Disc_Time_s = res$disc_time,
          stringsAsFactors = FALSE)
      }
      cat("\n")
    }
  }
  
  list(results      = do.call(rbind, rows),
       acc_matrices = acc_matrices,   # named list: ct -> matrix
       auc_matrices = auc_matrices,
       # keep old names pointing to NB for backward compat with make_table2/3
       acc_matrix   = acc_matrices[["nb"]],
       auc_matrix   = auc_matrices[["nb"]])
}


# =============================================================================
# make_table3 — accuracy + AUC table for EVERY classifier
# =============================================================================

make_table3 <- function(bench1) {
  
  clf_labels <- c(nb = "NaiveBayes", lr = "LogisticRegFactor",
                  xgb = "XGBoost",   oner = "OneR")
  
  for (ct in names(bench1$acc_matrices)) {
    lbl      <- clf_labels[ct]
    acc_mat  <- bench1$acc_matrices[[ct]]
    auc_mat  <- bench1$auc_matrices[[ct]]
    
    # ── Accuracy table ───────────────────────────────────────────────────────
    t_acc <- as.data.frame(round(acc_mat, 4))
    t_acc$Best <- apply(acc_mat, 1, function(r) {
      if (all(is.na(r))) return(NA_character_)
      paste(names(which(r == max(r, na.rm = TRUE))), collapse = ", ")
    })
    mean_acc <- round(colMeans(acc_mat, na.rm = TRUE), 4)
    t_acc <- rbind(t_acc, c(as.list(mean_acc), Best = "—"))
    t_acc <- cbind(Dataset = c(rownames(acc_mat), "Mean"), t_acc)
    rownames(t_acc) <- NULL
    
    tag <- paste0("table3_acc_", ct)
    cat(sprintf("\n=== Table 3 [%s] — Accuracy per discretiser ===\n", lbl))
    print(t_acc, row.names = FALSE)
    write.csv(t_acc, paste0(tag, ".csv"), row.names = FALSE)
    
    # ── AUC table ────────────────────────────────────────────────────────────
    if (!is.null(auc_mat)) {
      t_auc <- as.data.frame(round(auc_mat, 4))
      t_auc$Best <- apply(auc_mat, 1, function(r) {
        if (all(is.na(r))) return(NA_character_)
        paste(names(which(r == max(r, na.rm = TRUE))), collapse = ", ")
      })
      mean_auc <- round(colMeans(auc_mat, na.rm = TRUE), 4)
      t_auc <- rbind(t_auc, c(as.list(mean_auc), Best = "—"))
      t_auc <- cbind(Dataset = c(rownames(auc_mat), "Mean"), t_auc)
      rownames(t_auc) <- NULL
      
      tag_auc <- paste0("table3_auc_", ct)
      cat(sprintf("\n=== Table 3 [%s] — AUC per discretiser ===\n", lbl))
      print(t_auc, row.names = FALSE)
      write.csv(t_auc, paste0(tag_auc, ".csv"), row.names = FALSE)
    }
  }
}


# =============================================================================
# make_table4 — Friedman + Holm-Wilcoxon for EVERY classifier × metric
# =============================================================================
make_table4 <- function(bench1) {
  
  clf_labels <- c(nb = "NaiveBayes", lr = "LogisticRegFactor",
                  xgb = "XGBoost",   oner = "OneR")
  
  all_results <- list()
  
  for (ct in names(bench1$acc_matrices)) {
    lbl <- clf_labels[ct]
    
    # Accuracy
    res_acc <- run_posthoc(bench1$acc_matrices[[ct]], "Accuracy", lbl)
    if (!is.null(res_acc)) {
      pw <- res_acc$pairwise
      names(pw) <- c("vs Method", "Mean GOLD", "Mean Other",
                     "Mean Diff", "p (raw)", "p (Holm)", "Sig.")
      tag <- paste0("table4_posthoc_acc_", ct, ".csv")
      write.csv(pw, tag, row.names = FALSE)
      cat(sprintf("  → saved %s\n", tag))
      all_results[[paste0(ct, "_acc")]] <- res_acc
    }
    
    # AUC
    if (!is.null(bench1$auc_matrices[[ct]])) {
      res_auc <- run_posthoc(bench1$auc_matrices[[ct]], "AUC", lbl)
      if (!is.null(res_auc)) {
        pw <- res_auc$pairwise
        names(pw) <- c("vs Method", "Mean GOLD", "Mean Other",
                       "Mean Diff", "p (raw)", "p (Holm)", "Sig.")
        tag <- paste0("table4_posthoc_auc_", ct, ".csv")
        write.csv(pw, tag, row.names = FALSE)
        cat(sprintf("  → saved %s\n", tag))
        all_results[[paste0(ct, "_auc")]] <- res_auc
      }
    }
  }
  
  invisible(all_results)
}

run_posthoc <- function(mat, metric_label, clf_label) {
  
  mat_cc <- mat[complete.cases(mat), , drop = FALSE]
  if (nrow(mat_cc) < 5 || ncol(mat_cc)<2L) {
    cat(sprintf("\n  [%s | %s] Not enough complete datasets (%d).\n",
                clf_label, metric_label, nrow(mat_cc)))
    return(invisible(NULL))
  }
  
  cat(sprintf("\n%s\n", strrep("=", 65)))
  cat(sprintf(" Post-hoc: %s | %s  (%d datasets)\n",
              clf_label, metric_label, nrow(mat_cc)))
  cat(sprintf("%s\n", strrep("=", 65)))
  
  ft <- friedman.test(mat_cc)
  cat(sprintf("\nFriedman: chi2(%.0f) = %.4f, p = %.4f  %s\n",
              ft$parameter, ft$statistic, ft$p.value,
              ifelse(ft$p.value < 0.05, "[SIGNIFICANT]", "[not significant]")))
  
  if (!"GOLD" %in% colnames(mat_cc)) return(invisible(NULL))
  
  competitors <- setdiff(colnames(mat_cc), "GOLD")
  pvals <- vapply(competitors, function(nm)
    tryCatch(
      wilcox.test(mat_cc[, "GOLD"], mat_cc[, nm],
                  paired = TRUE, exact = FALSE)$p.value,
      error = function(e) NA_real_),
    numeric(1))
  adj  <- p.adjust(pvals, method = "holm")
  diff <- vapply(competitors, function(nm)
    mean(mat_cc[, "GOLD"] - mat_cc[, nm], na.rm = TRUE), numeric(1))
  
  pw <- data.frame(
    vs          = competitors,
    Mean_GOLD   = round(mean(mat_cc[, "GOLD"], na.rm = TRUE), 4),
    Mean_Other  = round(vapply(competitors, function(nm)
      mean(mat_cc[, nm], na.rm = TRUE), numeric(1)), 4),
    Mean_Diff   = round(diff,  4),
    p_raw       = pvals,
    p_Holm      = adj,
    Sig         = ifelse(adj < 0.05, "YES *", "no"),
    stringsAsFactors = FALSE)
  pw <- pw[order(pw$p_Holm), ]
  
  cat("\nHolm-corrected Wilcoxon: GOLD vs each discretiser\n\n")
  print(pw, row.names = FALSE)
  invisible(list(friedman = ft, pairwise = pw, matrix = mat_cc))
}




# =============================================================================
# PART 8 — COMPARISON 2: GOLD vs raw — accuracy and AUC
# =============================================================================

run_comparison2 <- function(datasets,k=5,seed=42) {
  shared <- c(if(has_pkg("nnet")) "lr",if(has_pkg("xgboost")) "xgb")
  labels <- c(nb="NaiveBayes",gnb="GaussianNB",lr="LogisticReg",xgb="XGBoost")
  rows <- list()
  cat("\nCOMPARISON 2 — GOLD vs raw features\n")
  for(nm in names(datasets)) {
    dat <- datasets[[nm]]
    a <- tryCatch(cv_run(dat,gold,c("nb",shared),k=k,seed=seed,
                         continuous=FALSE,track_bins=FALSE),
                  error=function(e) {message(nm," GOLD: ",conditionMessage(e));NULL})
    b <- tryCatch(cv_run(dat,no_disc,c("gnb",shared),k=k,seed=seed,
                         continuous=TRUE,track_bins=FALSE),
                  error=function(e) {message(nm," raw: ",conditionMessage(e));NULL})
    if(is.null(a) || is.null(b)) next
    row <- list(Dataset=nm)
    for(pair in list(c("nb","gnb","NB"),c("lr","lr","LogisticReg"),
                     c("xgb","xgb","XGBoost"))) {
      d <- pair[1]; r <- pair[2]; label <- pair[3]
      if(!d %in% names(a$accuracy) || !r %in% names(b$accuracy)) next
      row[[paste0("Disc_",label)]] <- unname(a$accuracy[d])
      row[[paste0("None_",if (r == "gnb") "GNB" else label)]] <- unname(b$accuracy[r])
      row[[paste0("Diff_",label)]] <- unname(a$accuracy[d]-b$accuracy[r])
      row[[paste0("AUC_Disc_",label)]] <- unname(a$auc[d])
      row[[paste0("AUC_None_",label)]] <- unname(b$auc[r])
      row[[paste0("AUC_Diff_",label)]] <- unname(a$auc[d]-b$auc[r])
    }
    rows[[length(rows)+1L]] <- as.data.frame(row,stringsAsFactors=FALSE)
    cat(nm,": complete\n")
  }
  if(!length(rows)) stop("No datasets completed comparison 2")
  do.call(rbind,rows)
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
    p_raw      = pvals,
    p_Holm     = adj,
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


make_table5 <- function(bench2) {
  has_lr  <- "Disc_LogisticReg" %in% names(bench2)
  has_xgb <- "Disc_XGBoost"     %in% names(bench2)
  col_order <- c("Dataset","Disc_NB","None_GNB","Diff_NB")
  if(has_lr)  col_order <- c(col_order,"Disc_LogisticReg","None_LogisticReg","Diff_LR")
  if(has_xgb) col_order <- c(col_order,"Disc_XGBoost","None_XGBoost","Diff_XGB")
  t5 <- bench2[,intersect(col_order,names(bench2)),drop=FALSE]
  num_cols  <- setdiff(names(t5),"Dataset")
  t5[num_cols] <- lapply(t5[num_cols], function(x) round(as.numeric(x),4))
  diff_cols <- grep("^Diff_",names(t5),value=TRUE)
  acc_cols  <- setdiff(num_cols,diff_cols)
  mean_row  <- c(list(Dataset="Mean"),
                 setNames(lapply(acc_cols,  function(c) round(mean(t5[[c]],na.rm=TRUE),4)), acc_cols),
                 setNames(lapply(diff_cols, function(c) round(mean(t5[[c]],na.rm=TRUE),4)), diff_cols))
  t5 <- rbind(t5, as.data.frame(mean_row,stringsAsFactors=FALSE))
  cat("\n=== Table 5: GOLD vs None accuracy ===\n")
  print(t5, row.names=FALSE);  write.csv(t5,"table5_gold_vs_none.csv",row.names=FALSE)
  invisible(t5)
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
  cat("\n=== Table 6a: Wilcoxon — Accuracy: GOLD vs None ===\n")
  print(t6_acc, row.names=FALSE)
  write.csv(t6_acc,"table6a_wilcoxon_acc.csv",row.names=FALSE)
  
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
# Table 7 — Runtime comparison across discretisers
# =============================================================================
make_table7 <- function(bench1) {
  
  # ── 1. Per-dataset runtime (summed over all classifiers → same value per clf)
  rt <- unique(bench1$results[, c("Dataset", "Discretizer", "Disc_Time_s")])
  
  # ── 2. Wide matrix: rows = datasets, cols = discretisers
  rt_wide <- reshape(rt,
                     idvar   = "Dataset",
                     timevar = "Discretizer",
                     direction = "wide")
  names(rt_wide) <- gsub("Disc_Time_s\\.", "", names(rt_wide))
  rownames(rt_wide) <- rt_wide$Dataset
  
  # ── 3. Summary stats per discretiser
  disc_cols <- setdiff(names(rt_wide), "Dataset")
  t7 <- do.call(rbind, lapply(disc_cols, function(nm) {
    v <- as.numeric(rt_wide[[nm]])
    data.frame(
      Discretizer = nm,
      Mean_s      = round(mean(v, na.rm = TRUE), 3),
      Median_s    = round(median(v, na.rm = TRUE), 3),
      Min_s       = round(min(v,  na.rm = TRUE), 3),
      Max_s       = round(max(v,  na.rm = TRUE), 3),
      SD_s        = round(sd(v,   na.rm = TRUE), 3),
      stringsAsFactors = FALSE
    )
  }))
  t7 <- t7[order(t7$Mean_s), ]
  rownames(t7) <- NULL
  
  cat("\n=== Table 7: Discretiser runtime (total seconds across CV folds, per dataset) ===\n")
  print(t7, row.names = FALSE)
  write.csv(t7, "table7_runtime.csv", row.names = FALSE)
  
  # ── 4. Statistical tests on runtime
  rt_mat <- as.matrix(rt_wide[, disc_cols])
  rt_mat_cc <- rt_mat[complete.cases(rt_mat), , drop = FALSE]
  
  if (nrow(rt_mat_cc) >= 5) {
    cat("\n--- Runtime statistical tests ---\n")
    
    ft <- friedman.test(rt_mat_cc)
    cat(sprintf(
      "\nFriedman test: chi2(%.0f) = %.4f, p = %.4f  %s\n",
      ft$parameter, ft$statistic, ft$p.value,
      ifelse(ft$p.value < 0.05, "[SIGNIFICANT]", "[not significant]")
    ))
    
    if ("GOLD" %in% colnames(rt_mat_cc)) {
      competitors <- setdiff(colnames(rt_mat_cc), "GOLD")
      pvals <- vapply(competitors, function(nm)
        tryCatch(
          wilcox.test(rt_mat_cc[, "GOLD"], rt_mat_cc[, nm],
                      paired = TRUE, exact = FALSE)$p.value,
          error = function(e) NA_real_
        ), numeric(1))
      adj  <- p.adjust(pvals, method = "holm")
      diff <- vapply(competitors, function(nm)
        mean(rt_mat_cc[, "GOLD"] - rt_mat_cc[, nm], na.rm = TRUE),
        numeric(1))
      
      pw7 <- data.frame(
        vs          = competitors,
        Mean_GOLD   = round(mean(rt_mat_cc[, "GOLD"], na.rm = TRUE), 3),
        Mean_Other  = round(vapply(competitors, function(nm)
          mean(rt_mat_cc[, nm], na.rm = TRUE), numeric(1)), 3),
        Mean_Diff_s = round(diff, 3),
        p_raw       = pvals,
        p_Holm      = adj,
        Sig         = ifelse(adj < 0.05, "YES *", "no"),
        stringsAsFactors = FALSE
      )
      pw7 <- pw7[order(pw7$p_Holm), ]
      
      cat("\nPost-hoc Wilcoxon (Holm-corrected): GOLD runtime vs each discretiser\n\n")
      print(pw7, row.names = FALSE)
      write.csv(pw7, "table7b_runtime_posthoc.csv", row.names = FALSE)
    }
  }
  invisible(list(summary = t7, matrix = rt_wide))
}


# =============================================================================
# RUN — serial; all outputs in their own folder to preserve previous experiments
# =============================================================================
GOLD_RUN_BENCHMARK <- FALSE
GOLD_K_FOLDS <- 5L
GOLD_SEED <- 42L
GOLD_N_SHUTTLE <- 4000L
GOLD_OUTPUT_DIR <- "./GOLD_results"

if (isTRUE(GOLD_RUN_BENCHMARK)) {
  need("discretization"); need("mlbench"); need("nnet")
  need("xgboost"); need("pROC")
  source('GOLD_tests.R', local=TRUE)
  stopifnot(gold_self_test())
  dir.create(GOLD_OUTPUT_DIR,showWarnings=FALSE,recursive=TRUE)
  oldwd <- getwd()
  setwd(GOLD_OUTPUT_DIR)
  tryCatch({
    cat("Loading datasets...\n")
    loaded <- load_datasets(n_shuttle=GOLD_N_SHUTTLE)
    datasets <- loaded$datasets; meta <- loaded$meta
    datasets <- datasets[names(datasets)!="Shuttle"]
    meta     <- meta[names(meta)!="Shuttle"]
    
    cat("Loaded ",length(datasets)," datasets\n",sep="")
    t1 <- make_table1(datasets,meta)
    methods <- build_disc_methods()
    stopifnot(identical(names(methods),c("GOLD","CAIM",
                                         #"CACC",
                                         "Ameva",
                                         "ChiMerge","Chi2","ExtendChi2","MDLP")))
    
    bench1 <- run_comparison1(datasets, methods, k=GOLD_K_FOLDS,seed=GOLD_SEED)
    saveRDS(bench1,"./bench1.rds")
    write.csv(bench1$results,"bench1_results.csv",row.names=FALSE)
    make_table2(bench1)
    make_table3(bench1)
    make_table4(bench1)
    bench2 <- run_comparison2(datasets,k=GOLD_K_FOLDS,seed=GOLD_SEED)
    saveRDS(bench2,"./bench2.rds")
    write.csv(bench2,"./bench2_results.csv",row.names=FALSE)
    make_table5(bench2)
    make_table6(bench2)
    make_table7(bench1)
    cat("\nBenchmark finished; outputs are in ",normalizePath(getwd()),"\n",sep="")
  },finally=setwd(oldwd))
}
