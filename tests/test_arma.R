# Script to verify unit tests with respect to DSE
# Run with R version 2.14.1 (2011-12-22)
# Copyright (C) 2011 The R Foundation for Statistical Computing
# ISBN 3-900051-07-0
# Platform: x86_64-pc-linux-gnu (64-bit)

library('dse')

# For the simulation unittest
AR <- array(c(1, .5, .3, 0, .2, .1, 0, .2, .05, 1, .5, .3), c(3,2,2))
MA <- array(c(1, .2, 0, .1, 0, 0, 1, .3), c(2,2,2))
X <- array(c(1, .3, 0, .05, 0, 0.1, 1, .3), c(2,2,2))
arma <- ARMA(A=AR, B=MA, C=NULL)

sampleT <- 10
setRNG(seed=0)
noise <- makeTSnoise(sampleT, 2, 2)

# Results for test_simulate unittest
R_result <- simulate(arma, noise=noise, sampleT=sampleT)

TREND <- c(1,2)
arma_trend <- ARMA(A=AR, B=MA, C=NULL, TREND=TREND)
R_result_trend <- simulate(arma_trend, noise=noise, sampleT=sampleT)

arma_trend_ext <- ARMA(A=AR, B=MA, C=X, TREND=TREND)
input0 <- array(c(0.1, 0.2, 0.15, 0.05), dim=c(2,2))
input <- array(c(0.1*c(1:10), 0.05*c(1:10)), dim=c(10,2))
R_result_trend_ext <- simulate(arma_trend_ext,
                               noise=noise,
                               sampleT=sampleT,
                               input0=input0,
                               input=input)

# Results for the forecast unittest
arma <- l(arma, R_result)
R_pred <- arma$estimates$pred

arma_trend <- l(arma_trend, R_result_trend)
R_pred_trend <- arma_trend$estimates$pred

# Results for the negloglike unittest
R_negloglike <- residualStats(R_pred, R_result$output)$like[1]

# Results for the forecast with horizon unittest
setRNG(seed=0)
sampleT <- 15
noise <- makeTSnoise(sampleT, 2, 2)
AR <- array(c(1, 0.3, 0.5),c(3,1,1))
MA <- array(c(1, 0.1), c(2,1,1))
TREND <- c(1:20)
arma <- ARMA(A=AR, B=MA, TREND=TREND)
truth <- simulate(arma, noise=noise, sampleT=sampleT)
forecast_obj <- forecast(obj=arma, data=truth, horizon=5)
forecast <- rbind(forecast_obj$pred, forecast_obj$forecast[[1]])

# A parameter estimation test
AR = array(c(1, 0.3, 0.5),c(3,1,1))
MA = array(c(1, 0.1), c(2,1,1))
arma <- ARMA(A=AR, B=MA)
simu <- simulate(arma, sampleT=sampleT, noise=noise)

AR = array(c(1, 0.01, 0.01),c(3,1,1))
MA = array(c(1, 0.01), c(2,1,1))
arma <- ARMA(A=AR, B=MA)
arma <- fixConstants(arma)
arma <- estMaxLik(obj1=arma, obj2=simu)

#tfplot(arma)
