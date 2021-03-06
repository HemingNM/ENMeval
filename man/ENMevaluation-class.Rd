\name{ENMevaluation-class }
\Rdversion{1.1}
\docType{class}
\alias{ENMevaluation-class}
\alias{ENMevaluation}
\title{Class \code{"ENMevaluation"}}
\description{ Objects of this class are generated by a call of \code{\link{ENMevaluate}}. }
\section{Objects from the Class}{
Objects can be created by calls of the form \code{new("ENMevaluation", ...)}.
}
\section{Slots}{
  \describe{
    \item{\code{algorithm}:}{Object of class \code{"character"}. The algorithm used for the analysis.}
    \item{\code{results}:}{Object of class \code{"data.frame"}. The full results table.}
    \item{\code{predictions}:}{Object of class \code{"RasterStack"}. Model predictions in geographic space.}
    \item{\code{models}:}{List of objects of class \code{"maxnet"} from the \pkg{maxnet} package or \code{"MaxEnt"} from the \pkg{dismo} package (depending on which algorithm was used). For \code{"Maxnet"}, see \pkg{maxnet} package documentation for more information. For \code{"MaxEnt"}, each of these entries include slots for lambda values and the original Maxent results table.  See \pkg{dismo} package documentation for more information.}
    \item{\code{partition.method}:}{Object of class \code{"character"}. Indicates the method used for data partitioning.}
    \item{\code{occ.pts}:}{Object of class \code{"data.frame"}. The original presence coordinates.}
    \item{\code{occ.grp}:}{Object of class \code{"numeric"}. The evaluation bin assignment for each occurrence point.}
    \item{\code{bg.pts}:}{Object of class \code{"data.frame"}. The background coordinates used for analysis.}
    \item{\code{bg.grp}:}{Object of class \code{"numeric"}. The evaluation bin assignment for each background point.}
    \item{\code{overlap}:}{Object of class \code{"matrix"}. Niche overlap statistic between models of different settings.}
  }
}
\author{ Jamie M. Kass <jkass@gc.cuny.edu> and Robert Muscarella <bob.muscarella@gmail.com> }
\examples{
showClass("ENMevaluation")
}
\keyword{classes}
