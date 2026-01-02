# dynamics.py: funciones

Este archivo implementa la dinámica en S1 (circulo) con drift de atencion y MLP,
y los integradores numericos.

## Clases de datos

- `MLPParams`: contiene los parametros del MLP (matrices `a`, `omega` y `activation`).
- `SimulationConfig`: parametros de simulacion (beta, dt, pasos, modo de atencion,
  integrador, etc.).
- `MLPConfig`: parametros para muestrear el MLP (numero de unidades, escala,
  activacion, y si el potencial esta atado).

## Funciones

- `_activation(z, activation)`: aplica la funcion de activacion (ReLU o GELU) a
  un array `z`.

- `sample_theta0(rng, n_particles)`: genera condiciones iniciales `theta` con
  distribucion uniforme en [0, 2pi).

- `sample_mlp_params(rng, config)`: muestrea los parametros del MLP (vectores
  `a` normalizados y `omega` escalado) y devuelve un `MLPParams`. En este codigo
  se impone que el MLP sea un gradiente (tie_potential=True).

- `attention_drift(theta, beta, mode, exclude_self=True)`: calcula la deriva de
  auto-atencion para cada particula en angulo. Usa pesos tipo softmax en
  `beta * cos(diff)`, con version normalizada o no normalizada. En modo
  unnormalized, el factor global puede ser `exp(row_max)` o `exp(row_max - beta)`
  segun `unnormalized_scale_mode`.

- `attention_drift_field(theta_eval, theta_particles, beta, mode)`: evalua el
  campo de deriva de atencion en puntos arbitrarios `theta_eval`, dados los
  angulos de las particulas (mismo `unnormalized_scale_mode` que arriba).

- `mlp_drift(theta, params)`: calcula el aporte del MLP a la deriva angular.
  Proyecta el campo del MLP en la direccion tangente del circulo.

- `simulate(theta0, sim_config, mlp_params, progress=None, progress_every=None)`:
  integra la dinamica durante `num_steps` y devuelve tiempos e historial de
  posiciones `theta` guardadas cada `save_every`.

- `_total_drift(theta, sim_config, mlp_params)`: suma el drift de atencion y, si
  existe, el drift del MLP.

- `step_theta(theta, sim_config, mlp_params)`: avanza un paso de tiempo usando
  el integrador configurado (`euler`, `rk2`, `rk4`) y devuelve `theta` modulo 2pi.

- `gamma_k_s1(beta, k_values)`: calcula `gamma_k` para S1 usando Bessel
  modificadas (`I_k`), usado en los plots de estabilidad.
