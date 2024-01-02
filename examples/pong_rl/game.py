'''
Define the game environment for the Pong RL example.
'''

import keyboard
from random import choice
from time import sleep
import tkinter

class Game:
    
    def __init__(self, model, train=False):
        self.model = model
        self.train = train

        if self.train:
            self.left_paddle = Paddle(type='wall')
            self.right_paddle = Paddle(type='ai')
        else:
            self.left_paddle = Paddle(type='player')
            self.right_paddle = Paddle(type='ai')

        self.ball = Ball()
        self.width = 400
        self.height = 250
        self.left_score = 0
        self.right_score = 0

    def run(self):
        # When training the AI, the game will be simulated instead of displayed

        # If not training, create and display the game canvas
        if not self.train:
            # Create the game canvas
            self.root = tkinter.Tk()
            self.root.title("Pong")
            self.root.configure(background='black')
            self.canvas = tkinter.Canvas(self.root, width=self.width, height=self.height)
            self.canvas.configure(background='black')
            self.root.after(100, self.update)
            self.root.mainloop()
        # If training, simulate running the actual game
        else:
            while True:
                self.update()

    def update(self):
        # Update the paddle and ball positions
        self.left_paddle.update(self.ball, self.model)
        self.right_paddle.update(self.ball, self.model)
        self.ball.update()

        # Check for ceiling / floor collisions
        if self.ball.y <= -(self.height / 2) + self.ball.size or self.ball.y >= (self.height / 2) - self.ball.size:
            self.ball.vy *= -1

        # Check for paddle collisions
        if self.ball.x <= -(self.width / 2) + self.ball.size + self.left_paddle.width:
            if self.left_paddle.y - self.left_paddle.height / 2 <= self.ball.y <= self.left_paddle.y + self.left_paddle.height / 2:
                self.ball.vx *= -1
            else:
                self.right_score += 1
                self.ball = Ball()

        if self.ball.x >= (self.width / 2) - self.ball.size - self.right_paddle.width:
            if self.right_paddle.y - self.right_paddle.height / 2 <= self.ball.y <= self.right_paddle.y + self.right_paddle.height / 2:
                self.ball.vx *= -1
            else:
                self.left_score += 1
                self.ball = Ball()

        # Make sure paddles don't go off screen
        for paddle in [self.left_paddle, self.right_paddle]:
            if paddle.y - paddle.height / 2 <= -(self.height / 2):
                paddle.y = -(self.height / 2) + paddle.height / 2
            elif paddle.y + paddle.height / 2 >= self.height / 2:
                paddle.y = self.height / 2 - paddle.height / 2

        if not self.train:
            sleep(0.002)

            # Clear canvas
            self.canvas.delete('all')

            # Draw paddles and ball
            left_paddle_coords = (self.left_paddle.x_offset, self.height / 2 - self.left_paddle.height / 2 + self.left_paddle.y,
                                    self.left_paddle.x_offset + self.left_paddle.width, self.height / 2 + self.left_paddle.height / 2 + self.left_paddle.y)
            right_paddle_coords = (self.width - self.right_paddle.x_offset - self.right_paddle.width, self.height / 2 - self.right_paddle.height / 2 + self.right_paddle.y,
                                        self.width - self.right_paddle.x_offset, self.height / 2 + self.right_paddle.height / 2 + self.right_paddle.y)
            ball_coords = (self.width / 2 - self.ball.size / 2 + self.ball.x, self.height / 2 - self.ball.size / 2 + self.ball.y,
                            self.width / 2 + self.ball.size / 2 + self.ball.x, self.height / 2 + self.ball.size / 2 + self.ball.y)
            self.canvas.create_rectangle(left_paddle_coords, fill='white')
            self.canvas.create_rectangle(right_paddle_coords, fill='white')
            self.canvas.create_rectangle(ball_coords, fill='white')

            # Draw the scores
            self.canvas.create_text(self.width/2, 15, text=f'{self.left_score} | {self.right_score}', fill='white', font=('Arial', 12))

            self.canvas.pack()
            
            # End game or continue updating state
            if keyboard.is_pressed('esc'):
                self.root.destroy()
            self.root.after(1, self.update)

class Paddle:

    def __init__(self, type):
        self.type = type

        self.y = 0
        self.height = 50
        self.width = 10
        self.x_offset = 10

    def update(self, ball, model):
        # Move paddle based on keyboard input
        if self.type == 'player':
            if keyboard.is_pressed('up'):
                self.y -= 1
            elif keyboard.is_pressed('down'):
                self.y += 1

        # Always matches ball height - Unbeatable, used for training
        elif self.type == 'wall':
            self.y = ball.y
        
        # Get model prediction, move paddle accordingly
        elif self.type == 'ai':
            movement = round(model.predict([ball.x, ball.y, ball.vx, ball.vy, self.y]).item())
            self.y += -1 if movement else 1

class Ball:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.vx = choice([-1, 1])
        self.vy = choice([-1, 1])
        self.size = 10

    def update(self):
        self.x += self.vx
        self.y += self.vy
