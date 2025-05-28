class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        #define the derivative of the function
        d = lambda x: 2*x
        #initialize x
        x = init
        for i in range(iterations):
            x = x - learning_rate * d(x)
        #limit x to 5 decimal places
        x = round(x, 5)
        return x