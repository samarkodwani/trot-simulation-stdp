"""
PROJECT: Static Quadruped CPG (Baseline)

ENGINEERING NOTES & STABILITY ANALYSIS:
During development, the code exploded (values went to infinity) with:
1) No random noise
2) dt = 0.5
3) w = 10

Understanding the cause:
- Without noise, it still worked but became completely sync-ed.
- The main culprit: dt = 0.5, as the voltage went to 1e159 (infinity)!
  For a stiff model like Izhikevich, a smaller timestep is required to catch the spike.
"""

import numpy as np
import matplotlib.pyplot as plt

class Quadruped:
    def __init__(self, N=4):
        self.N = N
        self.a = 0.02
        self.b = 0.2
        self.c = -65
        self.d = 8
        
        # self.v = np.ones(N) * self.c 
        # (After running) Possible crash, as everyone starts to spike together
        self.v = self.c + np.random.rand(N) * 10 # Added random noise
        self.u = self.v * self.b
        self.w = 5
        
        # --- GAIT PATTERN SETUP ---
        
        # A wrong approach. We actually have a pattern:
        '''
        self.W = np.zeros((N, N))
        self.W[0, 1] = self.w
        self.W[0, 2] = self.w
        ...
        '''

        # For Trot:
        '''
        self.W = np.zeros((N,N))
        pairs = [(0,1), (2,3), (0,2), (1,3)]  # This is the pattern for trot
        for i,j in pairs:
          self.W[i,j] = self.w
          self.W[j,i] = self.w
        '''

        # For Pace:
        self.W = np.zeros((N,N))
        pairs = [(0,1), (0,3), (2,1), (2,3)]
        for i, j in pairs:
            self.W[i, j] = self.w
            self.W[j, i] = self.w

        # Creating N^2 synapse conductances vs creating N conductances:
        # We add "weight" to the total "g bucket" of an individual neuron, which reduces complexity
        self.g = np.zeros(N)
        self.tau = 10

    def step(self, dt, I):
        E = -80
        I_syn = -self.g * (self.v - E)
        self.v += (0.04 * self.v**2 + 5 * self.v - self.u + 140 + I + I_syn) * dt
        self.u += self.a * (self.b * self.v - self.u) * dt

        fired = self.v >= 30
        # The above one is a mask; we need indices
        fired_meow = np.where(fired)[0]
        
        for i in fired_meow:
            self.g += self.W[:, i] # Check i-th column, it has all its connections
        
        self.v[fired] = self.c
        self.u[fired] += self.d
        self.g += (-self.g / self.tau) * dt # Update the conductance, at every snapshot
        return fired

# --- Simulation ---
T = 1000
dt = 0.1  # Better for a stiff model like this than 0.5 ms (why?)
steps = int(T / dt)
q = Quadruped()
trace = []
I = 5

for i in range(steps):
    q.step(dt, I)
    trace.append(q.v.copy())

trace = np.array(trace)

# --- Visualization ---
plt.figure(figsize=(10, 5))
# plt.plot(trace)
'''
plt.title("4 Neurons Firing (Uncoupled)")
plt.xlabel("Time (steps)")
plt.ylabel("Voltage (mV)")
plt.show()
'''

# Gait Visualization
colors = ['r', 'b', 'g', 'k']
names = ['FL', 'FR', 'HL', 'HR']

for i in range(len(colors)):
    current = trace[:, i]
    plt.plot(current + i*120, color=colors[i], label=names[i]) # Keeping them at different heights
    
plt.title("Quadruped Gait Pattern (Pace)") 
plt.xlabel("Time (steps)")
plt.yticks([]) # Hide the fake y-axis numbers
plt.legend(loc='upper right') # Show the labels
plt.tight_layout()
plt.show()