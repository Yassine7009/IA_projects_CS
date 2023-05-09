from cmath import inf
from nn import Parameter
import nn
import numpy as np


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """

        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        result = self.run(x)
        if nn.as_scalar(result) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        accu = False
        while accu == False:
            count = 0
            for x_data, y_data in dataset.iterate_once(1):
                if self.get_prediction(x_data) != nn.as_scalar(y_data):
                    self.get_weights().update(x_data, nn.as_scalar(y_data))
                    count += 1
            if count == 0:
                accu = True
        return self.get_weights()


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # 1st layer :
        self.w1 = nn.Parameter(1, 25)
        self.b1 = nn.Parameter(1, 25)
        # 2nd layer :
        self.w2 = nn.Parameter(25, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # 1st layer
        x1 = nn.Linear(x, self.w1)
        x2 = nn.AddBias(x1, self.b1)
        x3 = nn.ReLU(x2)
        # 2nd layer :
        x4 = nn.Linear(x3, self.w2)
        predicted_y = nn.AddBias(x4, self.b2)

        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SquareLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float("inf")
        while loss > 0.02:
            grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(
                self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)),
                [self.w1, self.b1, self.w2, self.b2],
            )
            self.w1.update(grad_w1, -0.01)
            self.b1.update(grad_b1, -0.01)
            self.w2.update(grad_w2, -0.01)
            self.b2.update(grad_b2, -0.01)
            loss = nn.as_scalar(
                self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            )


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # 1st layer :
        self.w1 = nn.Parameter(784, 300)
        self.b1 = nn.Parameter(1, 300)
        # 2nd layer :
        self.w2 = nn.Parameter(300, 140)
        self.b2 = nn.Parameter(1, 140)

        # 3rd layer
        self.w3 = nn.Parameter(140, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # 1st layer
        x1 = nn.Linear(x, self.w1)
        x2 = nn.AddBias(x1, self.b1)
        x3 = nn.ReLU(x2)
        # 2nd layer :
        x4 = nn.Linear(x3, self.w2)
        x5 = nn.AddBias(x4, self.b2)
        x6 = nn.ReLU(x5)

        # 3rd layer
        x7 = nn.Linear(x6, self.w3)
        pred = nn.AddBias(x7, self.b3)

        return pred

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accu = 0
        while accu < 0.97:
            # change after
            for x, y in dataset.iterate_once(60):
                grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = nn.gradients(
                    self.get_loss(x, y),
                    [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3],
                )
                self.w1.update(grad_w1, -0.4)
                self.b1.update(grad_b1, -0.4)
                self.w2.update(grad_w2, -0.4)
                self.b2.update(grad_b2, -0.4)
                self.w3.update(grad_w3, -0.4)
                self.b3.update(grad_b3, -0.4)
            accu = dataset.get_validation_accuracy()
            print(accu)


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # need to create one W, one Whidden
        # init
        self.W = nn.Parameter(self.num_chars, 100)
        self.b = nn.Parameter(1, 100)

        self.W2 = nn.Parameter(100, 100)
        self.b2 = nn.Parameter(1, 100)

        # hidden
        self.Wh = nn.Parameter(100, 100)
        self.bh = nn.Parameter(1, 100)

        self.Wh2 = nn.Parameter(100, 100)
        self.bh2 = nn.Parameter(1, 100)

        # final
        self.Wf = nn.Parameter(100, 5)
        self.bf = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        for i in range(len(xs)):
            if i == 0:
                # 1st layer
                z1 = nn.Linear(xs[i], self.W)
                z2 = nn.ReLU(nn.AddBias(z1, self.b))
                # 2nd layer
                z3 = nn.Linear(z2, self.W2)
                h = nn.ReLU(nn.AddBias(z3, self.b2))
            else:
                # 1st layer
                z1 = nn.AddBias(
                    nn.Add(nn.Linear(xs[i], self.W), nn.Linear(h, self.Wh)), self.bh
                )
                a1 = nn.ReLU(z1)

                z2 = nn.AddBias(nn.Linear(a1, self.Wh2), self.bh2)
                h = nn.ReLU(z2)

                # 2nd layer

        h = nn.AddBias(nn.Linear(h, self.Wf), self.bf)
        return h

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(xs), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accu = 0

        while accu < 0.82:
            for l in range(10):
                # change after
                for x, y in dataset.iterate_once(60):

                    (
                        grad_W,
                        grad_Wh,
                        grad_Wf,
                        grad_b,
                        grad_bf,
                        grad_bh,
                        grad_W2,
                        grad_b2,
                        grad_Wh2,
                        grad_bh2,
                    ) = nn.gradients(
                        self.get_loss(x, y),
                        [
                            self.W,
                            self.Wh,
                            self.Wf,
                            self.b,
                            self.bf,
                            self.bh,
                            self.W2,
                            self.b2,
                            self.Wh2,
                            self.bh2,
                        ],
                    )
                    self.W.update(grad_W, -0.15)
                    self.Wh.update(grad_Wh, -0.15)
                    self.Wf.update(grad_Wf, -0.15)
                    self.b.update(grad_b, -0.15)
                    self.bf.update(grad_bf, -0.15)
                    self.bh.update(grad_bh, -0.15)

                    self.W2.update(grad_W2, -0.15)
                    self.b2.update(grad_b2, -0.15)
                    self.Wh2.update(grad_Wh2, -0.15)
                    self.bh2.update(grad_bh2, -0.15)
            accu = dataset.get_validation_accuracy()
            print(accu)
