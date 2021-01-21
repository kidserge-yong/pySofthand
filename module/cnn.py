import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import time
import imageio

class cnnRegession():
    def __init__(self, input_n = 1, output_n = 1):
        torch.manual_seed(1)    # reproducible
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_n, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, output_n),
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    def train(self, x, y, BATCH_SIZE = 64, EPOCH = 10):
        """
        x data (Variable), shape=(sample, channel)
        y data (Variable), shape=(sample, channel)
        """

        torch_dataset = Data.TensorDataset(x, y)

        loader = Data.DataLoader(
            dataset=torch_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=0,)

        fig, ax = plt.subplots(figsize=(16,10))
        my_images = []
        
        print('Training...')

        t0 = time.time()

        for epoch in range(EPOCH):

            for step, (batch_x, batch_y) in enumerate(loader): # for each training step

                # Progress update every 1 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(loader), elapsed))
                
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)

                prediction = self.net(b_x)     # input x and predict based on x

                loss = self.loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

                self.optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                self.optimizer.step()        # apply gradients

            if epoch % 10 == 0:
                elapsed = self.__format_time(time.time() - t0)
                print('Epoch {:} / {:}, Elapsed: {:}, Loss = {:}.'.format(epoch+1, EPOCH, elapsed, loss.data.numpy()))


            plt.cla()
            ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
            ax.set_xlabel('Independent variable', fontsize=24)
            ax.set_ylabel('Dependent variable', fontsize=24)
            ax.set_xlim(-11.0, 13.0)
            ax.set_ylim(-1.1, 1.2)
            #ax.scatter(b_x.data.numpy(), b_y.data.numpy(), color = "blue", alpha=0.2)
            #ax.scatter(b_x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
            ax.scatter(self.net(b_x).data.numpy(), b_y.data.numpy(), color = "blue", alpha=0.2)
            ax.scatter(self.net(b_x).data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
            ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
                    fontdict={'size': 24, 'color':  'red'})
            ax.text(8.8, -0.95, 'Loss = %.4f' % loss.data.numpy(),
                    fontdict={'size': 24, 'color':  'red'})
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            my_images.append(image)
        
        imageio.mimsave('./model/curve_2_model_3_batch.gif', my_images, fps=12)

        fig, ax = plt.subplots(figsize=(16,10))
        plt.cla()
        ax.set_title('Regression Analysis - model 3, Batches', fontsize=35)
        ax.set_xlabel('Independent variable', fontsize=24)
        ax.set_ylabel('Dependent variable', fontsize=24)
        ax.set_xlim(-11.0, 13.0)
        ax.set_ylim(-1.1, 1.2)
        ax.scatter(self.net(x).data.numpy(), y.data.numpy(), color = "blue", alpha=0.2)
        prediction = self.net(x)     # input x and predict based on x
        ax.scatter(self.net(x).data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
        plt.savefig('./model/curve_2_model_3_batches.png')
        plt.show()

    def __format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        import datetime
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def predict(self, x):
        return self.net(x)

    def save_pretrained(self, path=r"./model/cnnRegression"):
        torch.save(self.net, path)

    def load_pretrained(self, path=r"./model/cnnRegression"):
        model = torch.load(PATH)
        model.eval()



            

def plot(graph):
    plt.plot(graph)
    plt.show() 

if __name__ == "__main__":
    x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
    y = torch.sin(x) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1) 

    x, y = Variable(x), Variable(y)
    plt.figure(figsize=(10,4))
    plt.scatter(x.data.numpy(), y.data.numpy(), color = "blue")
    plt.title('Regression Analysis')
    plt.xlabel('Independent varible')
    plt.ylabel('Dependent varible')
    plt.show()

    cnn = cnnRegession()
    cnn.train(x, y, BATCH_SIZE = 64, EPOCH = 1)

    cnn.save_pretrained("./model/cnnRegression")