#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


class BrutalOptimLine():

  def __init__(self,z1, y1, dict_var, n_bins, vbase = 500, sbase = 100):

    self.z1                   = z1
    self.y1                   = y1
    self.gama                 = np.sqrt(z1*y1) #1/km
    self.n_bins               = n_bins
    self.dict_var             = dict_var
    self.vbase                = vbase #kV
    self.sbase                = sbase #MW
    self._history             = None
    
    self.loop()


  def history(self, current_state):

    if self._history is None:
        self._history = np.array([current_state])

    else:
        self._history = np.vstack([self._history, current_state])

  def loop(self, start_state = True):
    
    n_bins = self.n_bins
    
    vr_min, vr_max = self.dict_var['vr']
    L_min, L_max = self.dict_var['L(km)']
    sn_min, sn_max = self.dict_var['Sn']
    theta_min, theta_max = self.dict_var['theta']
    
    total_of_events = n_bins**4
    count = 0
    
    for vr in np.linspace(vr_min, vr_max, n_bins):
        for sn in np.linspace(sn_min, sn_max, n_bins):
            for theta in np.linspace(theta_min, theta_max, n_bins):
                for L in np.linspace(L_min, L_max, n_bins):
                    
                    out, Loss = self.compute(vr, L, theta, sn)
                    
                    if out:
                        norm_L = (L - L_min)/(L_max - L_min)
                        norm_vr = (vr - vr_min)/(vr_max - vr_min)
                        norm_sn = (sn - sn_min)/(sn_max - sn_min)
                        norm_theta = (theta - theta_min)/(theta_max - theta_min)
                        norm_Loss = 1 - Loss
                        
                        current_state = [L, theta, vr,sn, norm_Loss]
                        self.history(current_state)
                    
                    count += 1
                    print('(%.2f%%) concluido...'%(100*count/total_of_events), end = '\r')
                    
        

  def compute(self, vr, L, theta, sn):
    """ Verify whether the current state variables provide the Line a good operation point."""
    
    def calc_param(Zc,a,b,c,Vr,theta,pout=0):      

      Pn = (Vr)**2/np.real(Zc) #MW total
      #Teste com carga nominal ou sobrecarga
      if pout != 0:
        Pout = Pn*pout #MW por fase
        
        Vrp = Vr/np.sqrt(3)
        Ir = np.conjugate((Pout + 1j*Pout*np.tan(theta*np.pi/180))/(np.sqrt(3)*Vr)) #kA
        V, Is = np.array([[a,b],[c,a]])@np.array([Vrp,Ir]) #kV,kA por fase
        Loss = 3*(np.real(V*np.conjugate(Is)) - np.real(Vrp*np.conjugate(Ir))) #MW total (Todas as linhas)
                  
      #Teste de efeito Ferranti
      else:
        V = 1*self.vbase/a #kV
        Is = V*c #kA
        Loss = 3*(np.real(1*self.vbase*np.conjugate(Is)/np.sqrt(3))) #MW
      
      if not(self.dict_var['vr'][0] <= abs(np.sqrt(3)*V/self.vbase) <= self.dict_var['vr'][1]) or abs(np.angle(V, deg = True)) > 30 or Loss > Pn:
          return False, (Loss/Pn)
      return True, (Loss/Pn)

    Zc = np.sqrt(self.z1/self.y1) #Ohm
    a = np.cosh(self.gama*L) #1
    b = Zc*np.sinh(self.gama*L) #Ohm
    c = (1/Zc)*np.sinh(self.gama*L) #Mho
    Vr = vr*self.vbase #kV
    
    # Teste com carga nominal
    Teste1, perc_Loss = calc_param(Zc,a,b,c,Vr,theta,pout=sn)

    # Teste de efeito Ferranti
    #Teste2 = calc_param(Zc,a,b,c,Vr,theta,pout=0)
    
    
    return Teste1, perc_Loss #*Teste2

  def area(self, array_modules):

    ang_inc = 2*np.pi/array_modules.shape[0]
    array_angles = np.arange(np.pi/2, 5*np.pi/2 + ang_inc, ang_inc)[:-1]
        
    array_coord = np.array([[mod*np.cos(array_angles[idx]),mod*np.sin(array_angles[idx])] for idx, mod in enumerate(array_modules)])
    array_coord = np.vstack([array_coord, array_coord[0]])
    
    area = 0.5*np.sum(np.abs([np.linalg.det(array_coord[line:line+2]) for line in range(array_coord.shape[0] - 1)]))
    
    return area, array_coord[:-1]

  def plot(self, points, **kwg):
    
    
    if 'keys' in kwg:
        dict_keys = {'L(km)': 'L',
                     'theta': '$\Delta\\theta$',
                     'vr': '$V_r$',
                     'Sn': '$S_n$'}
        labels = kwg['keys']
        labels = [dict_keys[key] for key in list(dict_keys.keys())]
        labels.append('Loss')
    else:
        labels = points.shape[1]*['']
    if 'output_path' in kwg:
        output = kwg['output_path']
    else:
        output = ''
    
    if 'area' in kwg:
        area = kwg['area']
    else:
        area = ''
    
    if 'title' in kwg:
        title = kwg['title']
    else:
        title = 'Linha de transmissao'
    
    fig, ax = plt.subplots()
    
    # Plotar esqueleto
    n_points = points.shape[1]
    array_mod = np.linspace(0,1,5)[1:]
    array_angle =  np.arange(np.pi/2, 5*np.pi/2 + 2*np.pi/n_points, 2*np.pi/n_points)
    
    array_layers =  np.array([ [[mod*np.cos(angle), mod*np.sin(angle)] for angle in array_angle] for mod in np.linspace(0,1,5)[1:] ])
    [ax.plot(layer[:,0], layer[:,1], ':k', alpha = 0.5) for layer in array_layers]
    [ax.plot([0,layer1[0]],[0,layer1[1]], '-k', alpha = 0.5, linewidth = 0.5) for layer1 in array_layers[-1]]
    
    identity = np.identity(5)
    array_angle = np.linspace(np.pi/2, 5*np.pi/2 - 2*np.pi/n_points, n_points)
    # Plotting Polygon
    for layer in points:
        ax.scatter((np.identity(array_angle.shape[0])*np.cos(array_angle))@layer, (np.identity(array_angle.shape[0])*np.sin(array_angle))@layer)
        #print([array_angle[idx]*180/np.pi,np.cos(array_angle[idx]),np.cos(array_angle[idx])])
        #ax.plot(np.cos(array_angle[idx])*points[:,idx], np.sin(array_angle[idx])*points[:,idx])
        
    #plt.gca().add_patch(patches.Polygon(points, alpha = 0.5, lw = 1.5, ls = '-', edgecolor = 'k', facecolor = 'purple'))
    
    array_angle =  np.arange(np.pi/2, 5*np.pi/2 + 2*np.pi/n_points, 2*np.pi/n_points)
    # Adjusting text
    [ax.text(layer1[0]*(1 + 0.08/abs(layer1[0])), layer1[1]*(1 + 0.05/abs(layer1[1])), labels[idx],fontweight = 'medium', fontsize = 12) for idx, layer1 in enumerate(array_layers[-1][:-1])]
    ax.text(0.50, 0.85, 'Area: %.4f'%(area) , fontfamily = 'serif', fontweight = 'bold', fontsize = 12)
    ax.axis(False)
    plt.title(title, fontsize = 'xx-large', fontweight = 'bold', pad = 20)
    
    plt.show()
    
    


# In[13]:


variables = {'L(km)': [80,300],
             'theta': [-15,15],             
             'vr': [0.95,1.05],
             'Sn': [0.1,4]
             }

# keep non-simmetrical variables on the bottom

n_bins = 50


# In[ ]:


'''
# 500 kV
z1 = 0.022190408640634214 + 1j*0.34665038554475636
y1 = 1j*4.845131194232457e-6
Line1 = BrutalOptimLine(z1,y1,variables,n_bins)
keys = list(Line1.dict_var.keys())
np.save('Line1.npy', Line1._history)
print('salvou 1 \n\n\n')
#Line1.plot(Line1._history, area = 0.5, keys = keys, title = '500 kV')


# In[ ]:

# 500 kV compacta
z2 = 0.016627313705499122 + 0.2676290633843461*1j;
y2 = 1j*6.0975547780729794e-6
Line2 = BrutalOptimLine(z2,y2,variables,n_bins)
np.save('Line2.npy', Line2._history)
print('salvou 2 \n\n\n')
'''
# 500 kV capacitada
z3 = 0.016631216371080317 + 0.2499488173319771*1j
y3 = 1j*6.7068614020792455e-6
Line3 = BrutalOptimLine(z3,y3,variables,n_bins)
np.save('Line3.npy', Line3._history)
print('salvou 3 \n\n\n')

# 500 kV expandido
z4 = 0.01349074087148576 + 0.19432017869484816*1j
y4 = 3.0000000000000136e-9 + 1j*8.604206875650306e-6;
Line4 = BrutalOptimLine(z4,y4,variables,n_bins)
np.save('Line4.npy', Line3._history)
print('salvou 4 \n\n\n')

