import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

print("\t Water Level : [0-100] \t Water Use : [0-100]")
input_water_level, input_water_use = input("\tEnter Water level and Water use : ").split()
input_water_level = float(input_water_level)
input_water_use = float(input_water_use)

#set universal
water_level = np.arange(0,101,1)
water_use = np.arange(0,101,1)
dam_gate = np.arange(0,51,1)

# membership function
level_l = fuzz.trimf(water_level, [0,0,50])
level_m = fuzz.trimf(water_level, [0,50,100])
level_h = fuzz.trimf(water_level, [50,100,100])
use_l = fuzz.trimf(water_use, [0,0,50])
use_m = fuzz.trimf(water_use, [0,50,100])
use_h = fuzz.trimf(water_use, [50,100,100])
gate_l = fuzz.trimf(dam_gate, [0,0,25])
gate_m = fuzz.trimf(dam_gate, [0,25,50])
gate_h = fuzz.trimf(dam_gate, [25,50,50])

#set input 
water_level_l = fuzz.interp_membership(water_level, level_l, input_water_level)
water_level_m = fuzz.interp_membership(water_level, level_m, input_water_level)
water_level_h = fuzz.interp_membership(water_level, level_h, input_water_level)
water_use_l = fuzz.interp_membership(water_use, use_l, input_water_use)
water_use_m = fuzz.interp_membership(water_use, use_m, input_water_use)
water_use_h = fuzz.interp_membership(water_use, use_h, input_water_use)

#rule 
rule1 = np.fmax(water_level_h,water_use_h)
rule2 = np.fmax(water_level_h,water_use_m)
rule3 = np.fmax(water_level_m,water_use_h)
rule4 = np.fmax(water_level_m,water_use_m)
rule5 = np.fmax(water_level_m,water_use_l)
rule6 = np.fmax(water_level_l,water_use_m)
rule7 = np.fmax(water_level_l,water_use_l)

#alpha cut
alpha_h = np.fmin(rule1,gate_h)
alpha_m = np.fmin(rule2,gate_m)
alpha_m = np.fmin(rule3,alpha_m)
alpha_m = np.fmin(rule4,alpha_m)
alpha_m = np.fmin(rule5,alpha_m)
alpha_m = np.fmin(rule6,alpha_m)
alpha_l = np.fmin(rule7,gate_l)



# Show membership functions
fig, (a, b, c) = plt.subplots(nrows=3, figsize=(8, 9))

a.plot(water_level, level_l, 'b', linewidth=1, label='Low')
a.plot(water_level, level_m, 'g', linewidth=1, label='Medium')
a.plot(water_level, level_h, 'r', linewidth=1, label='High')
a.set_title('Water Volume')
a.legend()

b.plot(water_use, use_l, 'b', linewidth=1, label='Low')
b.plot(water_use, use_m, 'g', linewidth=1, label='Normal')
b.plot(water_use, use_h, 'r', linewidth=1, label='High')
b.set_title('Water Demand')
b.legend()

c.plot(dam_gate, gate_l, 'b', linewidth=1, label='Close')
c.plot(dam_gate, gate_m, 'g', linewidth=1, label='Normal')
c.plot(dam_gate, gate_h, 'r', linewidth=1, label='High')
c.set_title('Dam Gate Level')
c.legend()



# Aggregate all three output membership functions together
aggregated = np.fmax(alpha_h, alpha_m)
aggregated = np.fmax(alpha_l, aggregated)

# # Calculate defuzzified result
gate = fuzz.defuzz(dam_gate, aggregated, 'centroid')
gate_plot = fuzz.interp_membership(dam_gate, aggregated, gate) 
print('Gate Dam Level : ',gate)

zero_plot = np.zeros_like(gate)

fig, x = plt.subplots(figsize=(8, 9))
x.plot(dam_gate, gate_l, 'b', linewidth=0.5, linestyle='--', )
x.plot(dam_gate, gate_m, 'g', linewidth=0.5, linestyle='--')
x.plot(dam_gate, gate_h, 'r', linewidth=0.5, linestyle='--')
x.fill_between(dam_gate, zero_plot , aggregated, facecolor='pink')
x.plot([gate, gate], [0, gate_plot], 'k', linewidth=1.5)
x.set_title('result: line, and Graph alpha cut')

plt.show()



