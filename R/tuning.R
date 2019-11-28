#############################################
#########	TUNING FUNCTION	#############
#############################################
# THIS FUNCTION DOES SPATIALLY-INDEPENDENT EVALUATIONS
# INPUT ARGUMENTS COME FROM WRAPPER FUNCTION

tuning <- function (occ, env, bg.coords, occ.grp, bg.grp, method, algorithm, args,
                    args.lab, categoricals, aggregation.factor, kfolds, bin.output,
                    clamp, alg, rasterPreds, parallel, numCores, progbar, updateProgress,
                    userArgs,
                    threshold = 5, rand.percent = 50, iterations = 100 # pRoc
                    ) {

  # extract predictor variable values at coordinates for occs and bg
  pres <- as.data.frame(extract(env, occ))
  bg <- as.data.frame(extract(env, bg.coords))

  # if rows have NA for any predictor, toss them out
  # also redefine occ and bg without NA rows
  numNA.pres <- sum(rowSums(is.na(pres)) > 0)
  numNA.bg <- sum(rowSums(is.na(bg)) > 0)
  if (numNA.pres > 0) {
    message(paste("There are", numNA.pres, "occurrence records with NA for at least
                  one predictor variable. Removing these records from analysis,
                  resulting in", nrow(pres) - numNA.pres, "records..."))
    occ <- occ[!apply(pres, 1, anyNA),]
    pres <- pres[!apply(pres, 1, anyNA),]
  }
  if (numNA.bg > 0) {
    message(paste("There are", numNA.bg, "background records with NA for at least one predictor variable.
                  Removing these records from analysis, resulting in", nrow(bg) - numNA.bg, "records..."))
    bg.coords <- bg.coords[!apply(bg, 1, anyNA),]
    bg <- bg[!apply(bg, 1, anyNA),]
  }

  if (!is.null(categoricals)) {
    for (i in 1:length(categoricals)) {
      pres[, categoricals[i]] <- as.factor(pres[, categoricals[i]])
      bg[, categoricals[i]] <- as.factor(bg[, categoricals[i]])
    }
  }

  # assign partition groups based on choice of method
  if ("checkerboard1" %in% method) {
    method <- c(method = "checkerboard1", aggregation.factor = aggregation.factor)
    group.data <- get.checkerboard1(occ, env, bg.coords, aggregation.factor)
  }
  if ("checkerboard2" %in% method) {
    method <- c(method = "checkerboard2", aggregation.factor = aggregation.factor)
    group.data <- get.checkerboard2(occ, env, bg.coords, aggregation.factor)
  }
  if ("block" %in% method)
    group.data <- get.block(occ, bg.coords)
  if ("jackknife" %in% method)
    group.data <- get.jackknife(occ, bg.coords)
  if ("randomkfold" %in% method) {
    method <- c(method = "randomkfold", number.folds = kfolds)
    group.data <- get.randomkfold(occ, bg.coords, kfolds)
  }
  if ("user" %in% method) {
    method <- c(method= "user", number.folds = length(unique(occ.grp)))
    group.data <- get.user(occ.grp, bg.grp)
  }


  # define number of groups (the value of "k")
  nk <- length(unique(group.data$occ.grp))

  # differential behavior for parallel and default
  if (parallel == TRUE) {
    # set up parallel computing
    allCores <- detectCores()
    if (is.null(numCores)) {
      numCores <- allCores
    }
    c1 <- makeCluster(numCores)
    registerDoParallel(c1)
    numCoresUsed <- getDoParWorkers()
    message(paste("Of", allCores, "total cores using", numCoresUsed))
    #cat(paste("Of", allCores, "total cores using", numCoresUsed, "\n"))

    # log file to record status of parallel loops
    message("Running in parallel...")
    if (algorithm == 'maxnet') {
      # load maxnet package because it is not loaded automatically when embedded in foreach
      # if not loaded here, dismo::evaluate fails later
      # library(maxnet)
      out <- foreach(i = seq_len(length(args)),
                     .packages = c("dismo", "raster", "ENMeval", "maxnet")) %dopar% {
                       modelTune.maxnet(pres, bg, env, nk, group.data, args[[i]],
                                        rasterPreds=T, clamp, 
                                        occ = occ, threshold = threshold, # pRoc
                                        rand.percent = rand.percent, iterations = iterations) # pRoc
                     }
    } else if (algorithm == 'maxent.jar') {
      out <- foreach(i = seq_len(length(args)),
                     .packages = c("dismo", "raster", "ENMeval", "rJava")) %dopar% {
                       modelTune.maxentJar(pres, bg, env, nk, group.data, args[[i]],
                                           userArgs, rasterPreds=T, clamp, categoricals, 
                                           occ = occ, threshold = threshold, # pRoc
                                           rand.percent = rand.percent, iterations = iterations) # pRoc
                     }
    }
    # stopCluster(c1)
  } else {
    out <- list()
    if (progbar == TRUE & !is.function(updateProgress)) {
      pb <- txtProgressBar(0, length(args), style = 3)
    }
    for (i in 1:length(args)) {
      # set up the console progress bar (progbar),
      # or the shiny progress bar (updateProgress)
      if (length(args) > 1) {
        if (is.function(updateProgress)) {
          text <- paste0('Running ', args.lab[[1]][i], args.lab[[2]][i], '...')
          updateProgress(detail = text)
        } else if (progbar == TRUE) {
          setTxtProgressBar(pb, i)
        }
      }
      if (algorithm == 'maxnet') {
        out[[i]] <- modelTune.maxnet(pres, bg, env, nk, group.data, args[[i]],
                                     rasterPreds=T, clamp, 
                                     occ = occ, threshold = threshold, # pRoc
                                     rand.percent = rand.percent, iterations = iterations) # pRoc
      } else if (algorithm == 'maxent.jar') {
        out[[i]] <- modelTune.maxentJar(pres, bg, env, nk, group.data, args[[i]],
                                        userArgs, rasterPreds=T, clamp, categoricals, 
                                        occ = occ, threshold = threshold, # pRoc
                                        rand.percent = rand.percent, iterations = iterations) # pRoc
      }
    }
    if (progbar==TRUE) close(pb)
  }

  # gather all full models into list
  full.mods <- lapply(out, function(x) x[[1]])
  # gather all statistics into a data frame
  statsTbl <- as.data.frame(t(sapply(out, function(x) x[[2]])))
  # gather AICc
  # aiccTbl <- data.frame(t(sapply(out, function(x) x[[3]][,c(1,4)])))
  AICc <- sapply(out, function(x) x[[3]][,1])
  # AICc <- unlist(aiccTbl$AICc)
  delta.AICc <- (AICc - min(AICc, na.rm = TRUE))
  w.AIC <- (exp(-0.5 * delta.AICc))/(sum(exp(-0.5 * delta.AICc), na.rm = TRUE))
  aicc <- data.frame(AICc, delta.AICc, w.AIC, parameters = sapply(out, function(x) x[[3]][,4]))
  
  # if (rasterPreds) {
  #   predictive.maps <- stack(sapply(out, function(x) x[[3]]))
  # } else {
  #   predictive.maps <- stack()
  # }

  AUC.DIFF <- statsTbl[,1:nk]
  AUC.TEST <- statsTbl[,(nk+1):(2*nk)]
  OR10 <- statsTbl[,((2*nk)+1):(3*nk)]
  ORmin <- statsTbl[,((3*nk)+1):(4*nk)]
  pROC.ratio <- statsTbl[,((4*nk)+1):(5*nk)]
  pROC.p <- statsTbl[,((5*nk)+1):(6*nk)]
  # rename column fields
  names(AUC.DIFF) <- paste("diff.AUC_bin", 1:nk, sep = ".")
  Mean.AUC.DIFF <- rowMeans(AUC.DIFF)
  Var.AUC.DIFF <- corrected.var(AUC.DIFF, nk)
  names(AUC.TEST) <- paste("AUC_bin", 1:nk, sep = ".")
  Mean.AUC <- rowMeans(AUC.TEST)
  Var.AUC <- corrected.var(AUC.TEST, nk)
  names(OR10) <- paste("test.or10pct_bin", 1:nk, sep = ".")
  Mean.OR10 <- rowMeans(OR10)
  Var.OR10 <- apply(OR10, 1, var)
  names(ORmin) <- paste("test.orMTP_bin", 1:nk, sep = ".")
  Mean.ORmin <- rowMeans(ORmin)
  Var.ORmin <- apply(ORmin, 1, var)
  # names(pROC)
  Mean.pROC.ratio <- rowMeans(pROC.ratio, na.rm=T)
  Var.pROC.ratio <- apply(pROC.ratio, 1, var, na.rm=T)
  Mean.pROC.p <- rowMeans(pROC.p, na.rm=T)
  # get training AUCs for each model
  full.AUC <- double()

  for (i in 1:length(full.mods)) {
    if (algorithm == 'maxnet') {
      full.AUC[i] <- dismo::evaluate(pres, bg, full.mods[[i]])@auc
    } else if (algorithm == 'maxent.jar') {
      full.AUC[i] <- full.mods[[i]]@results[5]
    }
  }

  # get total number of parameters
  # nparam <- numeric()
  # for (i in 1:length(full.mods)) {
  #   if (algorithm == 'maxnet') {
  #     nparam[i] <- length(full.mods[[i]]$betas)
  #   } else if (algorithm == 'maxent.jar') {
  #     nparam[i] <- get.params(full.mods[[i]])
  #   }
  # }

  #  if (rasterPreds==TRUE) { # this should now work even if rasterPreds==F
  #     aicc <- calc.aicc(nparam, occ, predictive.maps)
  #  } else {
  #    aicc <- rep(NaN, length(full.AUC))
  #  }

  features <- args.lab[[1]]
  rm <- args.lab[[2]]
  settings <- paste(args.lab[[1]], args.lab[[2]], sep = "_")

  res <- data.frame(settings, features, rm, train.AUC = full.AUC,
                    avg.test.AUC = Mean.AUC, var.test.AUC = Var.AUC,
                    avg.diff.AUC = Mean.AUC.DIFF, var.diff.AUC = Var.AUC.DIFF,
                    
                    avg.test.pROC.ratio = Mean.pROC.ratio, var.test.pROC.ratio = Var.pROC.ratio,
                    avg.pROC.p = Mean.pROC.p,
                    
                    avg.test.orMTP = Mean.ORmin, var.test.orMTP = Var.ORmin,
                    avg.test.or10pct = Mean.OR10, var.test.or10pct = Var.OR10, aicc)
  if (bin.output == TRUE) {
    res <- as.data.frame(cbind(res, AUC.TEST, AUC.DIFF, OR10, ORmin))
  }

  # if (rasterPreds==TRUE) {
  #   names(predictive.maps) <- settings
  #   if(parallel) {
  #     message("predictions are not available when running in parallel")
      predictive.maps <- stack()
  #     raster::removeTmpFiles(h=0)
  #     stopCluster(c1)
  #   }# because predictive maps won't be available 
    # when running in parallel, it is better just to remove them from results
  # }

  results <- ENMevaluation(algorithm = alg, results = res, predictions = predictive.maps,
                           models = full.mods, partition.method = method,
                           occ.pts = occ, occ.grp = group.data[[1]],
                           bg.pts = bg.coords, bg.grp = group.data[[2]])
  if (parallel == TRUE) {stopCluster(c1)}
  return(results)
}
