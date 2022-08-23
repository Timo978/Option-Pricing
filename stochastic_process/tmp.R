library(plotly)
library(akima)
library(fields)

call_bsm = function (So,K,r,T,type,sig){
  d1 = (log(So/K) + (r+ (sig*sig)/2)*T)/(sig*sqrt(T))
  d2 = d1 - sig*sqrt(T)
  if (type == "Call")
  {price <- So*pnorm(d1) - K*exp(-r*T)*pnorm(d2)
  return (price)}
  else if (type == "Put")
  {price  <- -So*pnorm(-d1) + K*exp(-r*T)*pnorm(-d2)
  return (price)}
}

setwd('/Users/yangtianyu/Desktop/')
df <- read.csv('tmp.csv')

p <- plot_ly(x = df$t, y = df$k,
             z = matrix(df$iv,nrow=length(df$t))) %>% add_surface()
p

interpolate = interp(df$t, df$k, df$iv,duplicate="strip")
q <- plot_ly(x = interpolate$x, y = interpolate$y,
             z = interpolate$z)%>% add_surface()
q

dt = 1/252
dk = 0.5
deltac_by_deltat = c()
deltac_by_deltak = c()
delta2c_by_deltak2 = c()
local_volatility = c()
for (i in (1:length(df$t))) {
    deltac_by_deltat[i] = (call_bsm(So = 2.872, df$k[i],
        r = 0.0066, df$t[i] + dt, type = "Call", sig = df$iv[i]) -
        call_bsm(So = 2.872, df$k[i], r = 0.0066, df$t[i],
            type = "Call", sig = df$iv[i]))/(df$t[i] +
        dt - df$t[i])

    deltac_by_deltak[i] = (call_bsm(So = 2.872, df$k[i] + dk, r = 0.0066, df$t[i], type = "Call", sig = df$iv[i]) -
        call_bsm(So = 2.872, df$k[i], r = 0.0066, df$t[i],type = "Call", sig = df$iv[i]))/(df$k[i] + dk - df$k[i])

    delta2c_by_deltak2[i] = (call_bsm(So = 2.872, df$k[i] +
        dk, r = 0.0066, df$t[i], type = "Call", sig = df$iv[i]) -
        2 * call_bsm(So = 2.872, df$k[i], r = 0.0066, df$t[i],
            type = "Call", sig = df$iv[i]) + call_bsm(So = 2.872,
        df$k[i] - dk, r = 0.0066, df$t[i], type = "Call",
        sig = df$iv[i]))/(df$k[i] + dk - df$k[i])^2

    local_volatility[i] = sqrt((deltac_by_deltat[i] + 0.0066 *
        df$k[i] * deltac_by_deltak[i])/(0.5 * df$k[i] *
        df$k[i] * delta2c_by_deltak2[i]))

}
df$local_vol = local_volatility
df = df[!(df$local_vol >= 1), ]
head(df)