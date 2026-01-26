import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import time
import random
import gymnasium as gym
from gymnasium import spaces
import threading
from typing import Optional
import os

from popy.simulation_tools import MonkeyBanditTask
from popy.config import PROJECT_PATH_LOCAL


class GraphicalMonkeyBanditTask:
    """
    Graphical wrapper for the MonkeyBanditTask environment
    """
    def __init__(self, env=None):
        # Create or use the provided environment
        if env is None:
            from gymnasium.envs.registration import register
            try:
                gym.envs.registration.registry["MonkeyBandit-v0"]
            except KeyError:
                register(
                    id="MonkeyBandit-v0",
                    entry_point=MonkeyBanditTask
                )
            self.env = gym.make("MonkeyBandit-v0")
        else:
            self.env = env
        
        # Data storage
        self.data = pd.DataFrame(columns=['trial_id', 'block_id', 'best_arm', 'action', 'reward'])
        self.trial_count = 0
        
        # GUI setup
        self.root = tk.Tk()
        self.root.title("Monkey Bandit Task")
        self.root.geometry("600x400")
        
        # Control flow variables
        self.action_lock = threading.Event()
        self.user_action = None
        self.waiting_for_action = False
        
        # Button appearance
        self.default_button_bg = self.root.cget('bg')
        self.highlight_color = "yellow"
        
        self.create_ui()
        self.setup_keyboard_bindings()
    
    def create_ui(self):
        # Instructions
        tk.Label(self.root, text="Choose one button per trial to earn rewards", 
                 font=("Arial", 14)).pack(pady=20)
        
        # Feedback display
        self.feedback_label = tk.Label(self.root, text="", font=("Arial", 14), height=2)
        self.feedback_label.pack(pady=20)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=30)
        
        # Create 3 arm buttons
        self.buttons = []
        for i in range(self.env.n_arms):
            btn = tk.Button(button_frame, text=f"Button {i+1}", width=10, height=2,
                          command=lambda arm=i: self.on_button_click(arm))
            btn.pack(side=tk.LEFT, padx=15)
            self.buttons.append(btn)
        
        # Keyboard instructions
        key_instructions = tk.Label(self.root, text="You can also use arrow keys: ← (Left), ↑ (Up), → (Right)", 
                                   font=("Arial", 12))
        key_instructions.pack(pady=20)
        
        # Control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=20)
        
        self.start_button = tk.Button(control_frame, text="Start Task", command=self.start_task)
        self.start_button.pack(side=tk.LEFT, padx=15)
        
        self.quit_button = tk.Button(control_frame, text="Quit & Save Data", command=self.save_and_quit)
        self.quit_button.pack(side=tk.LEFT, padx=15)
    
    def setup_keyboard_bindings(self):
        self.root.bind('<Left>', lambda event: self.simulate_button_press(0) if self.waiting_for_action else None)
        self.root.bind('<Up>', lambda event: self.simulate_button_press(1) if self.waiting_for_action else None)
        self.root.bind('<Right>', lambda event: self.simulate_button_press(2) if self.waiting_for_action else None)
    
    def simulate_button_press(self, button_idx):
        """Simulates a physical button press for keyboard actions"""
        if self.waiting_for_action:
            # Visually press the button
            self.buttons[button_idx].config(relief=tk.SUNKEN, bg=self.highlight_color)
            self.root.update()
            
            # Trigger the button click action
            self.on_button_click(button_idx)
    
    def highlight_button(self, button_idx):
        """Highlight the selected button"""
        self.buttons[button_idx].config(bg=self.highlight_color)
        self.root.update()
    
    def reset_button_highlight(self, button_idx):
        """Reset button to default appearance"""
        self.buttons[button_idx].config(bg=self.default_button_bg, relief=tk.RAISED)
        self.root.update()
    
    def start_task(self):
        self.start_button.config(state=tk.DISABLED)
        observation, info = self.env.reset()
        self.trial_count = 0
        
        # Start the task in a separate thread
        threading.Thread(target=self.run_task, daemon=True).start()
    
    def run_task(self):
        while True:
            # Wait for user action
            self.waiting_for_action = True
            self.enable_buttons(True)
            self.feedback_label.config(text="Choose a button", fg="black")
            
            # Wait until user selects an action
            self.action_lock.wait()
            self.action_lock.clear()
            self.waiting_for_action = False
            
            if self.user_action is None:  # If quit was pressed
                break
                
            action = self.user_action
            self.enable_buttons(False)
            
            # Wait 500ms before showing outcome
            self.feedback_label.config(text="")
            time.sleep(0.5)  # Sleep for 500ms
            
            # Reset button highlight
            self.reset_button_highlight(action)
            
            # Execute the action in the environment
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Store data in pandas DataFrame
            new_data = {
                'trial_id': self.trial_count,
                'block_id': info['block_id'],
                'best_arm': info['best_arm'],
                'action': action,
                'reward': reward,
            }
            self.data = pd.concat([self.data, pd.DataFrame([new_data])], ignore_index=True)
            self.trial_count += 1

            print(new_data)
            
            # Update UI with the result
            if reward > 0:
                self.feedback_label.config(text="You won!\n1$", fg="green")
            else:
                self.feedback_label.config(text="No reward!\n0$", fg="red")
            
            # Wait 1 second before next trial
            self.root.update()
            time.sleep(1)  # Sleep for 1 second
            
            self.feedback_label.config(text="")  # Clear feedback
    
    def on_button_click(self, arm):
        if self.waiting_for_action:
            # For mouse clicks, we also highlight the button
            if not self.buttons[arm].cget('bg') == self.highlight_color:
                self.highlight_button(arm)
            
            self.user_action = arm
            self.action_lock.set()
    
    def enable_buttons(self, enable=True):
        state = tk.NORMAL if enable else tk.DISABLED
        for button in self.buttons:
            button.config(state=state)
    
    def save_and_quit(self):
        filename = "local_run.csv"
        
        floc = os.path.join(PROJECT_PATH_LOCAL, 'data', 'processed', 'behavior', filename)
        self.data.to_csv(floc, index=False)
        messagebox.showinfo("Data Saved", f"Data saved to {floc}")
        self.user_action = None
        self.action_lock.set()  # Release the lock if waiting
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()


# Example usage
if __name__ == "__main__":
    # Create the original environment
    env = MonkeyBanditTask(n_arms=3)
    
    # Create graphical wrapper
    app = GraphicalMonkeyBanditTask(env)
    app.run()