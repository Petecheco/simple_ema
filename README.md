# Simple Exponential Moving Average(simple_ema)

A simple exponential moving average class to update the model according to 
$$
\theta_{t} = \beta\cdot \theta_{t-1} + (1-\beta)\cdot \theta_{t}
$$


## Steps:
1. create the class, init $\beta$ and get a copy of the models parameters.
2. when calling **update()** method, and feed into a model, we operate the ema step according to the equation.
3. when calling **ema_inference()** method, we inference the output with the ema model.(without gradient) / or we can directly call ema.model
