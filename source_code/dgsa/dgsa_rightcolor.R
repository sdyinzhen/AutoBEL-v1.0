library(reshape)
library(ggplot2)
plotCDFS <- function (.clustering, .X, .code = NULL, .nBins = 3, .ggReturn = "plot", 
                      lwd = 1) 
{
  .cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                   "#0072B2", "#D55E00", "#CC79A7")
  .order <- order(.clustering)
  .clustering = .clustering[.order]
  .X = .X[.order,]
  if (.code == "all*") {
    .ggDATA <- rbind.data.frame(data.frame(.X, clustering = "prior"), 
                                data.frame(.X, clustering = as.character(.clustering)))
    .ggDATA$clustering <- factor(.ggDATA$clustering, levels = c(as.character(unique(.clustering)), 
                                                                "prior"))
    .ggDATA <- melt(.ggDATA, id = c("clustering"))
    .gg <- ggplot(.ggDATA) + stat_ecdf(aes(x = value, group = clustering, 
                                           colour = clustering), lwd = lwd) + ylab("Phi") + 
      ylim(c(0, 1)) + theme(legend.position = "top") + 
      xlab("parameter range") + scale_colour_manual(values = c(.cbbPalette[1:length(unique(.clustering))], 
                                                               "#000000")) + facet_wrap(~variable, scales = "free_x")
  }
  else {
    .params <- strsplit(.code, "|", fixed = TRUE)[[1]]
    if (length(.params) == 0 | length(.params) > 2) 
      stop("\".code\" is invalid!")
    .paramIndex <- apply(as.matrix(.params), 1, function(x) which(colnames(.X) %in% 
                                                                    x))
    if (length(.paramIndex) != length(.params)) 
      stop("\".code\" is invalid, one or both of the variables were not found!")
    if (length(.paramIndex) == 1) {
      .ggDATA <- rbind.data.frame(data.frame(parameter = .X[, 
                                                            .paramIndex], clustering = "prior"), data.frame(parameter = .X[, 
                                                                                                                           .paramIndex], clustering = as.character(.clustering)))
      .ggDATA$clustering <- factor(.ggDATA$clustering, 
                                   levels = c(as.character(unique(.clustering)), 
                                              "prior"))
      .gg <- ggplot(.ggDATA) + stat_ecdf(aes(x = parameter, 
                                             group = clustering, colour = clustering), lwd = lwd) + 
        xlab(colnames(.X)[.paramIndex]) + ylab("Phi") + 
        ylim(c(0, 1)) + theme(legend.position = "top") + 
        scale_colour_manual(values = c(.cbbPalette[1:length(unique(.clustering))], 
                                       "#000000"))
    }
    else {
      .brks = quantile(.X[, .paramIndex[2]], seq(0, 1, 
                                                 1/.nBins))
      if (length(unique(round(.brks, 8))) > 1) {
        .binning <- cut(.X[, .paramIndex[2]], breaks = .brks, 
                        include.lowest = TRUE, right = TRUE, labels = FALSE)
      }
      else {
        stop(paste("It appears that something is wrong with parameter: ", 
                   colnames(.X)[.paramIndex[2]]))
      }
      .ggDATA <- rbind.data.frame(data.frame(parameter = .X[, 
                                                            .paramIndex[1]], clustering = paste("Cluster_", 
                                                                                                .clustering, sep = ""), bin = .binning), data.frame(parameter = .X[, 
                                                                                                                                                                   .paramIndex[1]], clustering = paste("Cluster_", 
                                                                                                                                                                                                       .clustering, sep = ""), bin = "cluster_prior"))
      .ggDATA$bin <- factor(.ggDATA$bin, levels = c(unique(.binning), 
                                                    "cluster_prior"))
      .gg <- ggplot(.ggDATA) + stat_ecdf(aes(x = parameter, 
                                             group = bin, colour = bin), lwd = lwd) + xlab(.code) + 
        ylab("Phi") + ylim(c(0, 1)) + theme(legend.position = "top") + 
        facet_grid(~clustering) + scale_colour_manual(values = c((scales::hue_pal())(length(unique(.binning))), 
                                                                 "#000000"))
    }
  }
  .gg <- .gg + ggtitle(paste("CDFs for ", .code))
  if (.ggReturn == "plot") {
    print(.gg)
  }
  else {
    return(.gg)
  }
}


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

plotParetoDGSA <- function (.dgsa, .clusters = FALSE, .interaction = NULL, .hypothesis = TRUE, 
                            .ggReturn = FALSE) 
{
  .cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                   "#0072B2", "#D55E00", "#CC79A7")
  if (class(.dgsa) != "DGSAstructure") 
    stop("Passed object is not of class DGSAstructure. Exiting!")
  if (!is.null(.interaction)) {
    .paramIndex <- which(.dgsa$parameters == .interaction)
    if (length(.paramIndex) == 0) 
      stop("Parameter provided in \".interaction\" not found. Exiting!")
    .ggDATA <- t(.dgsa$sensitivityMatrix[, .paramIndex, ])
    .plotTitle <- paste("S(\"X\" | ", .dgsa$parameters[.paramIndex], 
                        ")", sep = "")
  }
  else {
    .ggDATA <- apply(.dgsa$sensitivityMatrix, 1, diag)
    .plotTitle <- "Main Sensitivities (marginal)"
  }
  colnames(.ggDATA) <- paste("Cluster", 1:ncol(.ggDATA), sep = "_")
  .ggDATA[is.nan(.ggDATA)] = 0
  .ggDATA <- as.data.frame(.ggDATA)
  .ggDATA$mean <- apply(.ggDATA, 1, mean)
  .ggDATA$parameters <- .dgsa$parameters
  .ggDATA <- .ggDATA[order(.ggDATA$mean), ]
  .levels <- .ggDATA$parameters
  .ggDATA <- melt(.ggDATA, id = c("parameters"))
  .ggDATA$parameters <- factor(.ggDATA$parameters, levels = .levels)
  .ggP <- ggplot(.ggDATA, aes(x = parameters, y = value,fill = variable,color = variable)) + 
    coord_flip() + 
    geom_bar(stat = "identity", position = "dodge", lwd = 0.2, colour = "black") + theme(legend.position = "bottom") + 
    geom_hline(yintercept = ifelse(.hypothesis, 1, NULL)) + scale_fill_manual(values=c(.cbbPalette[1:ncol(.ggDATA)],'darkred'))+
    scale_color_manual(values=c(.cbbPalette[1:ncol(.ggDATA)],'darkred'))+
    ggtitle(.plotTitle) 
  if (.ggReturn) {
    return(.ggP)
  }
  else {
    print(.ggP)
  }
}
