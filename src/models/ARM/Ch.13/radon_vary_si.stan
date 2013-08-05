data {
  int<lower=0> N; 
  vector[N] y;
  vector[N] x;
  int county[N];
} 
parameters {
  vector[85] a;
  vector[85] b;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  real mu_a;
  real<lower=0> sigma_b;
  real mu_b;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] <- a[county[i]] + b[county[i]] * x[i];
} 
model {
  mu_a ~ normal(0, .0001);
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);

  mu_b ~ normal(0, .0001);
  sigma_b ~ uniform(0, 100);
  b ~ normal (mu_b, sigma_b);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}