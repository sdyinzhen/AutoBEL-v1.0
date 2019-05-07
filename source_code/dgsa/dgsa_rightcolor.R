library(reshape)

dgsa <- function (.clusters, .X, .normalize = TRUE, .nBoot = 100, .interactions = FALSE, 
                  .nBins = 3, .alpha = 0.95, .parallel = FALSE, .progress = TRUE) 
{
  .order <- order(.clusters)
  .clusters = .clusters[.order]
  .X = .X[.order,]
  .l1 <- function(.PARAMETERS, .CLUSTERS) {
    .clusterCategories <- unique(.CLUSTERS)
    .clustCount <- apply(as.matrix(unique(.CLUSTERS)), 1, 
                         function(x) sum(.CLUSTERS == x))
    .N <- nrow(.PARAMETERS)
    .priors <- apply(.PARAMETERS, 2, function(x) quantile(x, 
                                                          seq(0, 1, 0.01)))
    .l1 <- matrix(NA, nrow = length(.clusterCategories), 
                  ncol = ncol(.PARAMETERS))
    for (i in 1:length(.clusterCategories)) {
      .l1[i, ] <- colSums(abs(.priors - apply(.PARAMETERS[.CLUSTERS == 
                                                            .clusterCategories[i], ], 2, function(x) quantile(x, 
                                                                                                              seq(0, 1, 0.01)))))
    }
    if (.normalize) {
      .bootMatrix <- array(NA, c(length(.clusterCategories), 
                                 ncol(.PARAMETERS), .nBoot))
      for (j in 1:.nBoot) {
        for (i in 1:length(.clusterCategories)) {
          .bootMatrix[i, , j] <- colSums(abs(.priors - 
                                               apply(.PARAMETERS[sample(.N, .clustCount[i], 
                                                                        replace = FALSE), ], 2, function(x) quantile(x, 
                                                                                                                     seq(0, 1, 0.01)))))
        }
      }
      .bootMatrix <- apply(.bootMatrix, c(1, 2), function(x) quantile(x, 
                                                                      .alpha))
      return(round(.l1, 9)/round(.bootMatrix, 9))
    }
    else {
      return(round(.l1, 9))
    }
  }
  .sensitivityMatrix <- array(NaN, c(length(unique(.clusters)), 
                                     ncol(.X), ncol(.X)))
  if (.progress) 
    print("Computing main effects...")
  .diagonal <- .l1(.X, .clusters)
  if (.interactions) {
    if (.progress) 
      print("Computing Interactions...")
    for (param in 1:ncol(.X)) {
      if (.progress) 
        print(paste("    Computing interactions conditioned on ", 
                    colnames(.X)[param], "...", sep = ""))
      for (clust in 1:length(unique(.clusters))) {
        .currentX <- .X[.clusters == clust, ]
        .brks = quantile(.currentX[, param], seq(0, 1, 
                                                 1/.nBins))
        if (length(unique(round(.brks, 8))) > 1 && anyDuplicated(.brks) == 
            0) {
          .binning <- cut(.currentX[, param], breaks = .brks, 
                          include.lowest = TRUE, right = TRUE, labels = FALSE)
          .sensitivityMatrix[clust, param, ] <- apply(.l1(.currentX, 
                                                          .binning), 2, mean)
        }
        else {
          if (.progress) 
            print(paste("        Computation failed due to inability to split  ", 
                        colnames(.X)[param], "  into ", .nBins, 
                        " bins.", sep = ""))
        }
      }
    }
  }
  for (i in 1:dim(.sensitivityMatrix)[1]) {
    diag(.sensitivityMatrix[i, , ]) <- .diagonal[i, ]
  }
  ret <- list(sensitivityMatrix = .sensitivityMatrix, parameters = colnames(.X))
  class(ret) <- "DGSAstructure"
  return(ret)
}

