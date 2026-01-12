#### Learning Trot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Quadruped:
    def __init__(self, N=4):
        self.N = N
        # a, b, c, and d are set for an RS (Regular Spiking) neuron 
        self.a = 0.02
        self.b = 0.2
        self.c = -65
        self.d = 8
        
        # Initialized with random noise so that all neurons don't fire at the same moment
        self.v = self.c + np.random.rand(N) * 10
        self.u = self.v * self.b
        
        # Initialize weak connections; no "self-connections"
        self.W = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    self.W[i, j] = np.random.rand()

        self.xtrace = np.zeros(N)
        self.ytrace = np.zeros(N) 

        self.g = np.zeros(N) 
        self.tau = 10.0
        self.lr = 0.01

    def step(self, dt, I):
        # The memory trace decays with time:
        taux = 20.0
        self.xtrace -= (self.xtrace / taux) * dt

        E = -80  # Inhibitory reversal potential
        I_syn = -self.g * (self.v - E) # Synaptic current
        
        self.v += (0.04 * self.v**2 + 5 * self.v - self.u + 140 + I + I_syn) * dt 
        self.u += self.a * (self.b * self.v - self.u) * dt

        fired = self.v >= 30  
        fired_indices = np.where(fired)[0] 

        for i in fired_indices:
            self.W[:, i] += self.lr * self.xtrace  # Increase the weight (Anti-Hebbian Learning)
            self.W[:, i] = np.clip(self.W[:, i], 0, 10)  # Clip weights so they don't explode or go negative
            self.W[i, i] = 0.0

            partner = 3 - i  # Force the 'partners' (diagonals) to not inhibit each other
            self.W[partner, i] = 0.0 

            self.g += self.W[i, :]
            self.xtrace[i] = 1.0

        self.v[fired] = self.c  # Reset the voltage
        self.u[fired] += self.d # Recovery variable increases
        self.g += (-self.g / self.tau) * dt # Conductance decay

        return fired

# Setup
T = 1000
dt = 0.1
steps = int(T / dt)
q = Quadruped()

# We teach the "brain" with manual input
teacher_signal = np.zeros((steps, 4))
period_steps = 1000

for i in range(steps):
    cycle_location = i % period_steps
    if cycle_location < (period_steps / 2):  # Train FL and HR for the first half of the cycle
        teacher_signal[i, 0] = 20
        teacher_signal[i, 3] = 20
    else:    # Train FR and HL for the second half of the cycle
        teacher_signal[i, 1] = 20
        teacher_signal[i, 2] = 20

trace = [] # Training run
for i in range(steps):
    I_now = teacher_signal[i]
    q.step(dt, I_now)
    trace.append(q.v.copy())

print("Training Complete.")

# Plotting the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(q.W, annot=True, cmap="Reds")
plt.show()

print("Starting Test Run...")
test_trace = []
T_test = 1000
steps_test = int(T_test / dt)
I_test = 10.0

for i in range(steps_test): # Running test
    q.step(dt, I_test)
    test_trace.append(q.v.copy())

test_trace = np.array(test_trace)
print("Test Complete.")

plt.figure(figsize=(10, 6))
colors = ['r', 'b', 'g', 'k']
names = ['FL', 'FR', 'HL', 'HR']

for i in range(4):  # Offset traces to make different legs visible
    current = test_trace[:, i]
    plt.plot(current + i*120, color=colors[i], label=names[i])

plt.yticks([])
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()
