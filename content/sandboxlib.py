import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ipywidgets import IntSlider
from matplotlib.widgets import Button, Slider
from ipywidgets import IntText, link
import ipywidgets as widgets

def abc(P, a, b, c, G=50.):
    """Das abc-Modell
    """
    # assert(a+b<=1)
    # assert(c<=1)
    # if a+b>1.:
    #     b = 1.-a
    df = pd.DataFrame(columns=["P","Q","Qd","Qb"], 
                      index=np.arange(len(P)), dtype="float")
    df["P"] = P

    if a+b>1.:
        return df
    if c>1:
        return df
    
    # Ergebniscontainer fuer den Gesamtabfluss
    Q = np.repeat(0.,len(P))
    Gw = np.repeat(0.,len(P))
    #G = np.repeat(0,len(P))
    for i in range(len(P)):
        df.loc[i,"Q"] = (1-a-b) * P[i] + c * G
        df.loc[i,"Qd"] = (1-a-b) * P[i]
        df.loc[i,"Qb"] = c * G
        G = (1-c)*G + a*P[i]
    return df
    #return pd.DataFrame({"P":P, "Q":Q, "Qb":Qb, "Qd":Qd}, index=t)

def plot_abc_model_old(P):
    t = np.arange(len(P))
    apar = 0.5
    bpar = 0.2
    cpar = 0.1
    sim = abc(P, apar, bpar, cpar)
    
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(num=" ")
    #fig.canvas.toolbar_position = 'bottom'
    fillQ = plt.fill_between(t, sim.Q, color="tab:blue", alpha=0.5)
    fillQb =plt.fill_between(t, sim.Qb, color="tab:blue", alpha=1.)
    line, = ax.plot(t, sim.Q, lw=1, color="black")
    ax.set_xlabel('Zeit [Tage]')
    plt.ylabel("Abfluss (mm/d]")
    plt.title("Abflusssimulation mit dem abc-Modell")
    
    
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(bottom=0.33)
    
    # Make a horizontal slider to control a.
    axa = fig.add_axes([0.1, 0.2, 0.65, 0.03])
    a_slider = Slider(
        ax=axa,
        label='a',
        valmin=0.,
        valmax=1.,
        valinit=apar,
    )
    
    # Make a horizontal slider to control b.
    axb = fig.add_axes([0.1, 0.1, 0.65, 0.03])
    b_slider = Slider(
        ax=axb,
        label='b',
        valmin=0.,
        valmax=1.,
        valinit=bpar,
    )
    
    # Make a horizontal slider to control b.
    axc = fig.add_axes([0.1, 0.0, 0.65, 0.03])
    c_slider = Slider(
        ax=axc,
        label='c',
        valmin=0.,
        valmax=1.,
        valinit=cpar,
    )
    
    # The function to be called anytime a slider's value changes
    # def update(val):
    #     line.set_ydata(abc(P, a_slider.val, b_slider.val, c_slider.val).Q)
    #     fig.canvas.draw_idle()
    
    def update(val): 
        sim = abc(P, a_slider.val, b_slider.val, c_slider.val)
        line.set_ydata(sim.Q)
        #optional preventing autoscaling of y-axis 
        ax.autoscale(False)
        #create invisible dummy object to extract the vertices 
        dummyQ = plt.fill_between(t, sim.Q , alpha=0)
        dpQ = dummyQ.get_paths()[0]
        dummyQ.remove()
        fillQ.set_paths([dpQ.vertices])
        
        dummyQb = plt.fill_between(t, sim.Qb , alpha=0)
        dpQb = dummyQb.get_paths()[0]
        dummyQb.remove()
        #update the vertices of the PolyCollection
        fillQb.set_paths([dpQb.vertices])
        fig.canvas.draw_idle()
    
    
    # register the update function with each slider
    a_slider.on_changed(update)
    b_slider.on_changed(update)
    c_slider.on_changed(update)
    
    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    #resetax = fig.add_axes([0.8, -0.2, 0.1, 0.04])
    #button = Button(resetax, 'Reset', hovercolor='0.975')
    #def reset(event):
    #    a_slider.reset()
    #    b_slider.reset()
    #    c_slider.reset()
    #button.on_clicked(reset)
    plt.show()


def plot_abc_model_old2(P):
    t = np.arange(len(P))+0.5
    apar = 0.5
    bpar = 0.2
    cpar = 0.1
    sim = abc(P, apar, bpar, cpar)
    
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(num=" ")
    #fig.canvas.toolbar_position = 'bottom'
    fillQ = plt.bar(t, sim.Q, color="tab:blue", alpha=0.5, width=1., label="Gesamtabfluss")
    fillQb =plt.bar(t, sim.Qb, color="tab:blue", alpha=1., width=1., label="Basisabfluss")
    #line, = ax.plot(t, sim.Q, lw=1, color="black")
    ax.set_xlabel('Zeit [z.B. Monate]')
    plt.ylabel("Abfluss (mm/Monat]")
    plt.title("Abflusssimulation mit dem abc-Modell")
    plt.grid()
    plt.xlim(0,len(P))
    plt.ylim(0,60)
    
    
    def makepartext(apar,bpar,cpar):
        return "a=%.2f\nb=%.2f\nc=%.2f\n1-a-b=%.2f\n" % (apar, bpar, cpar, 1-apar-bpar)
        
    partxt = plt.text(plt.xlim()[-1]*0.97, plt.ylim()[-1]*0.97, 
                       makepartext(apar,bpar,cpar), ha="right", va="top" )
    
    # Secondary axis
    Ncolor="grey"
    ax2=plt.twinx()
    fillN = plt.bar(t, -sim.P, color=Ncolor, alpha=0.5, width=1., label="Niederschlag")
    minyval = -400
    plt.ylim(minyval,0)
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    ticks = np.linspace(plt.ylim()[0],plt.ylim()[-1],len(labels))
    labels = -ticks
    labels[-1] = 0
    labels = labels.astype("int").astype("str")
    #ticks = ax2.get_yticks()
    #ax2.set_yticklabels( [] )
    ax2.set_yticks(ticks, labels=labels)
    plt.ylabel("Niederschlag (mm/Monat)", color=Ncolor)
    ax2.spines['bottom'].set_color(Ncolor)
    ax2.tick_params(axis='y', colors=Ncolor)
    
    lns = [fillN[0]]+[fillQ[0]]+[fillQb[0]]
    labs = ["Niederschlag","Gesamtabfluss","GW-Abfluss"]
    ax.legend(lns, labs, loc="upper center")
    
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(bottom=0.33)
    
    # Make a horizontal slider to control a.
    axa = fig.add_axes([0.1, 0.2, 0.65, 0.03])
    a_slider = Slider(
        ax=axa,
        label='a',
        valmin=0.,
        valmax=1.,
        valinit=apar,
    )
    
    # Make a horizontal slider to control b.
    axb = fig.add_axes([0.1, 0.1, 0.65, 0.03])
    b_slider = Slider(
        ax=axb,
        label='b',
        valmin=0.,
        valmax=1.,
        valinit=bpar,
    )
    
    # Make a horizontal slider to control b.
    axc = fig.add_axes([0.1, 0.0, 0.65, 0.03])
    c_slider = Slider(
        ax=axc,
        label='c',
        valmin=0.,
        valmax=1.,
        valinit=cpar,
    )
    
    # The function to be called anytime a slider's value changes
    # def update(val):
    #     line.set_ydata(abc(P, a_slider.val, b_slider.val, c_slider.val).Q)
    #     fig.canvas.draw_idle()
    
    def update(val): 
        sim = abc(P, a_slider.val, b_slider.val, c_slider.val)
        #line.set_ydata(sim.Q)
        #optional preventing autoscaling of y-axis 
        ax.autoscale(False)
        #create invisible dummy object to extract the vertices
        for i, rect in enumerate(fillQ):
            rect.set_height(sim.at[i,"Q"])
    
        for i, rect in enumerate(fillQb):
            rect.set_height(sim.at[i,"Qb"])
    
        partxt.set_text(makepartext(a_slider.val,b_slider.val,c_slider.val))
        if a_slider.val+b_slider.val>1:
            partxt.set_color("red")
        else:
            partxt.set_color("black")
    
    #    dummyQ = plt.bar(t, sim.Q , alpha=0)
    #    dpQ = dummyQ.get_paths()[0]
    #    dummyQ.remove()
    #    fillQ.set_paths([dpQ.vertices])
        
    #    dummyQb = plt.bar(t, sim.Qb , alpha=0)
    #    dpQb = dummyQb.get_paths()[0]
    #    dummyQb.remove()
    #    #update the vertices of the PolyCollection
    #    fillQb.set_paths([dpQb.vertices])
        fig.canvas.draw_idle()
    
    
    # register the update function with each slider
    a_slider.on_changed(update)
    b_slider.on_changed(update)
    c_slider.on_changed(update)
    
    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    #resetax = fig.add_axes([0.8, -0.2, 0.1, 0.04])
    #button = Button(resetax, 'Reset', hovercolor='0.975')
    #def reset(event):
    #    a_slider.reset()
    #    b_slider.reset()
    #    c_slider.reset()
    #button.on_clicked(reset)
    plt.show()


def plot_abc_model(P):
    t = np.arange(len(P))+0.5
    apar = 0.5
    bpar = 0.2
    cpar = 0.1
    sim = abc(P, apar, bpar, cpar)
    
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(num=" ", constrained_layout=True, figsize=(5,3))
    #fig.canvas.toolbar_position = 'bottom'
    fillQ = plt.bar(t, sim.Q, color="tab:blue", alpha=0.5, width=1., label="Gesamtabfluss")
    fillQb =plt.bar(t, sim.Qb, color="tab:blue", alpha=1., width=1., label="Basisabfluss")
    #line, = ax.plot(t, sim.Q, lw=1, color="black")
    ax.set_xlabel('Zeit [z.B. Monate]')
    plt.ylabel("Abfluss (mm/Monat]")
    plt.title("Abflusssimulation mit dem abc-Modell")
    plt.grid()
    plt.xlim(0,len(P))
    plt.ylim(0,60)
    
    
    def makepartext(apar,bpar,cpar):
        return "a=%.2f\nb=%.2f\nc=%.2f\n1-a-b=%.2f\n" % (apar, bpar, cpar, 1-apar-bpar)
        
    partxt = plt.text(plt.xlim()[-1]*0.97, plt.ylim()[-1]*0.97, 
                       makepartext(apar,bpar,cpar), ha="right", va="top" )
    
    # Secondary axis
    Ncolor="grey"
    ax2=plt.twinx()
    fillN = plt.bar(t, -sim.P, color=Ncolor, alpha=0.5, width=1., label="Niederschlag")
    minyval = -400
    plt.ylim(minyval,0)
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    ticks = np.linspace(plt.ylim()[0],plt.ylim()[-1],len(labels))
    labels = -ticks
    labels[-1] = 0
    labels = labels.astype("int").astype("str")
    #ticks = ax2.get_yticks()
    #ax2.set_yticklabels( [] )
    ax2.set_yticks(ticks, labels=labels)
    plt.ylabel("Niederschlag (mm/Monat)", color=Ncolor)
    ax2.spines['bottom'].set_color(Ncolor)
    ax2.tick_params(axis='y', colors=Ncolor)
    
    lns = [fillN[0]]+[fillQ[0]]+[fillQb[0]]
    labs = ["Niederschlag","Gesamtabfluss","GW-Abfluss"]
    ax.legend(lns, labs, loc="upper center")
    
    
    plt.show()
    
    def update(a, b, c): 
        sim = abc(P, a, b, c)
        #line.set_ydata(sim.Q)
        #optional preventing autoscaling of y-axis 
        ax.autoscale(False)
        #create invisible dummy object to extract the vertices
        for i, rect in enumerate(fillQ):
            rect.set_height(sim.at[i,"Q"])
    
        for i, rect in enumerate(fillQb):
            rect.set_height(sim.at[i,"Qb"])
    
        partxt.set_text(makepartext(a,b,c))
        if a+b>1:
            partxt.set_color("red")
        else:
            partxt.set_color("black")
        fig.canvas.draw_idle()
    
    aw = widgets.FloatSlider(value=0.5, min=0., max=1., step=0.05)
    bw = widgets.FloatSlider(value=0.2, min=0., max=1., step=0.05)
    cw = widgets.FloatSlider(value=0.1, min=0., max=1., step=0.05)
    _ = widgets.interact(update, 
                         a=aw,
                         b=bw,
                         c=cw)


    
# SCS-CN

cn2table = pd.read_csv("data/cn-table.csv", index_col="Landnutzung")
amctable = pd.read_csv("data/amc-table.csv", sep=";", index_col="cn")

def modify_cn(p5, vegetation, cn2):
    amc = "normal"
    if vegetation=="Ja":
        if p5<36:
            amc = "trocken"
        if p5>53:
            amc = "feucht"
    else:
        if p5<13:
            amc = "trocken"
        if p5>28:
            amc = "feucht"
    return amctable.at[cn2, amc]
    
def scscn(P, CN, r=0.2):

    P = P/25.4 # von mm in Inches
    S = (1000/CN) - 10
    Ia = r*S   # Anfangsverlust
    if P <= Ia:
        return 0
    Pe = ((P-Ia)**2 / (P-Ia+S)) * 25.4

    return Pe

def get_cn(lanu, soil, vegetation, p5):
    cn2 = cn2table.at[lanu,soil]
    #print("CN2=",cn2)
    CN = cn2*modify_cn(p5, vegetation, cn2)
    #print("CN=",CN)
    return CN
    
def make_txt(P, QD):
    if P==0:
        t = "P = %d mm\nQ$_D$ = %d mm\n" % (P, QD)
    else:
        t = "P = %d mm\nQ$_D$ = %d mm\n$\psi$ = %d Prozent" % (P, QD, 100*QD/P)
    return t


def plot_cn():
    fig, ax = plt.subplots(num=" ", constrained_layout=True, figsize=(5,3))
    CNs = [40, 50, 60, 70, 80, 90, 100]
    Ps = np.arange(0,101,1)
    for CN in CNs:
        Qs = np.array([scscn(P,CN) for P in Ps])
        plt.plot(Ps, Qs, color="black")
        plt.text(101, Qs[-1], "CN=%d" % CN, ha="left", va="center")
    CN = get_cn("Versiegelt", "A", "Ja", 0)
    Qs = [scscn(P, CN) for P in Ps]
    P=5
    Q = scscn(P, CN)
    line, = plt.plot(Ps, Qs, color="tab:red", lw=3)
    text = plt.text(5, 90, make_txt(P,Q), ha="left", va="top", color="tab:red")
    point, = plt.plot(P, Q, "ko", mfc="None", mec="tab:red", ms=10)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel("Niederschlag (mm)")
    plt.ylabel("Direktabfluss (mm)")
    plt.title("Das SCS-CN-Verfahren")
    
    plt.show()
    
    
    def update(Landnutzung, Boden, Vegetationsperiode, P5,P): 
        # print(Landnutzung)
        # print(Boden)
        # print(Vegetationsperiode)
        # print(P5)
        CN = get_cn(Landnutzung, Boden, Vegetationsperiode, P5)
        Qs = [scscn(P, CN) for P in Ps]
        Q = scscn(P, CN)
        line.set_ydata(Qs)
        point.set_xdata([P])
        point.set_ydata([Q])
        text.set_text(make_txt(P,Q))
        fig.canvas.draw_idle()
    
    _ = widgets.interact(update, Landnutzung=cn2table.index,
                         Boden=["A", "B", "C", "D"],
                         Vegetationsperiode=["Ja", "Nein"],
                         P5=[0,10,20,30,40,50,60],
                         P=(0,100,1))



def dm(init,v,ci,q,k,t):
    """Durchmischter Reaktor
    """
    return (init - (q*ci)/(k*v + q)) * np.exp((-(k + q/v)*t)) + (q*ci)/(k*v + q) 

def plot_dm(tmax=150):
    t = np.arange(tmax)
    
    # Parameter
    c0= 10  # in g/m3 
    Q = 1 * 3600 # von m3/s zu m3/h
    V = 1.e+5 # in m3
    cin= 1 # in g/m3
    k= 0.5 / 24 # von 1/d zu 1/h
    sim = dm(c0, V, cin, Q, k, t)
    
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(num=" ", constrained_layout=True, figsize=(5,3))
    #fig.canvas.toolbar_position = 'bottom'
    pltsim, = plt.plot(t, sim, color="tab:blue", lw=2., label="Konzentration")
    #line, = ax.plot(t, sim.Q, lw=1, color="black")
    ax.set_xlabel('Zeit (Stunden)')
    plt.ylabel("Konzentration (g/m³)")
    plt.title("Stoffkonzentration im durchmischten Reaktor")
    plt.grid()
    plt.xlim(0,t[-1])
    plt.ylim(0,10)
    
    
    def makepartext(c0, V, cin, Q, k):
        return "c$_0$=%.1f g/m³\nQ=%.1f m³/h\nV=%.0f m³\nc$_{in}$=%.1f g/m³\nk=%.2f 1/h\n" % (c0, Q, V, cin, k)
        
    partxt = plt.text(plt.xlim()[-1]*0.97, plt.ylim()[-1]*0.97, 
                       makepartext(c0, V, cin, Q, k), ha="right", va="top",
                     fontsize=10)
    
    #ax.legend()
    
    plt.show()
    
    def update(c0, V, cin, Q, k):
        sim2 = dm(c0, V, cin, Q, k, t)
        ax.autoscale(False)
        pltsim.set_ydata(sim2)
        partxt.set_text(makepartext(c0, V, cin, Q, k))
        fig.canvas.draw_idle()
    
    _ = widgets.interact(update, c0=(0,15,0.1),
                         V=(1e4,1e6,1000),
                         cin=(0.,10., 0.05),
                         Q=(1000,5000,100),
                         k=(0.0,0.1,0.001),)

    
# lanu_menu = Dropdown(
#     options=cn2table.index,
#     value = "Versiegelt",
#     description='Landnutzung',
# )

# soils_menu = Dropdown(
#     options=["A", "B", "C", "D"],
#     value = "A",
#     description='Boden',
# )
# saison_menu = Dropdown(
#     options=["Ja", "Nein"],
#     value = "Ja",
#     description='In der Vegetationsperiode?',
# )
# p5_menu = Dropdown(
#     options=[0,10,20,30,40,50,60],
#     value = 50,
#     description='Niederschlag der letzten 5 Tage (mm)',
# )



# -------------------------------------------------------------------------------------------------

#      
# Copyright (c) 2015, Peishi Jiang

#--------------------------------------------------------
# class GreenAmpt
#
#    __init__()
#    Fp
#    f
#    F
#    __EqnF
#    F_f
#
#--------------------------------------------------------
# This model is based on the Green-Ampt method described
# in <Applied Hydrology>, pp110.
# 
# Ref:
# Chow, Ven T., David R. Maidment, and Larry W. Mays. 
# Applied hydrology. 1988.
#--------------------------------------------------------

#from scipy.optimize import newton
#from pprint import pprint
#from json import dump
#from math import log
#import numpy as np

# class GreenAmpt(object):
#     '''
#     Green-Ampt Cumulative Infiltration 
#     '''
        
#     def __init__(self, K, dt, theta_i, theta_s, psi, i):
#         """
#         Constructor
#         """
#         self.K       = K                 # hydraulic conductivity
#         self.dt      = dt                # time resolution
#         self.theta_i = theta_i           # initial water content
#         self.theta_s = theta_s           # saturated water content
#         self.dtheta  = theta_s - theta_i # the change in the moisture content
#         self.psi     = psi               # wetting front soil suction head
#         if type(i) == list:
#             self.i = i                   # rainfall intensity
#         else:
#             self.i = [i]
    
#     def Fp(self, i):
#         """
#         Cumulative infiltration at the ponding time tp
#         """
#         return self.K * self.psi * self.dtheta / (i - self.K) 
    
#     def F(self, F_t, dt_t):
#         """
#         Solve Equation of Green-Ampt Cumulative Infiltration __EqnF
#         """
#         F_t_next = lambda F: self.__EqnF(F_t, dt_t, F)
#         return newton(F_t_next, 3)
    
#     def f(self, F):
#         """
#         Generate Green-Ampt Infiltration Rate at time t
#         """
#         if F == 0:
#             return -9999
#         else:
#             return self.K * (self.psi * self.dtheta / F + 1)    
        
#     def __EqnF(self, F_t, dt_t, F):
#         """
#         Equation of Green-Ampt Cumulative Infiltration after ponding
#         F:  Green-Ampt Cumulative Infiltration variable 
#         """
#         return F - F_t - self.K*dt_t - self.psi*self.dtheta*log((self.psi*self.dtheta+F)/(self.psi*self.dtheta+F_t))
    
#     def F_f(self):
#         """
#         Generate the time series of cumulative infiltration and infiltration rate
#         given the time series of rainfall intensity i
#         """
#         t_len = len(self.i)
#         F_all = []; f_all = []; t_all = []
#         # initial
#         F_all.append(0)
#         f_all.append(-9999)
#         t_all.append(0)
#         for ind in range(1, t_len+1):
#             i_t = self.i[ind-1]
#             f_t = f_all[ind-1]
#             F_t = F_all[ind-1]
#             if abs(f_t) <= i_t:
#                 # ponding occurs throught interval
#                 F_t_next = self.F(F_t, self.dt)
#                 f_t_next = self.f(F_t_next)
#             elif abs(f_t) > i_t:
#                 # no ponding at the beginning of the interval
#                 F_t_next_temp = F_t + i_t*self.dt
#                 f_t_next_temp = self.f(F_t_next_temp)
#                 if abs(f_t_next_temp) > i_t:
#                     # no ponding throughout interval
#                     f_t_next = f_t_next_temp
#                     F_t_next = F_t_next_temp
#                 elif abs(f_t_next_temp) <= i_t:
#                     # ponding occurs during interval
#                     Fp_t = self.Fp(i_t)
#                     print (i_t)
#                     dt_p = (Fp_t - F_t) / i_t
#                     F_t_next = self.F(Fp_t, self.dt - dt_p)
#                     f_t_next = self.f(F_t_next)        
#             F_all.append(F_t_next)
#             f_all.append(f_t_next)
#             t_all.append(self.dt*(ind))        
#         return F_all, f_all, t_all

# K = 5; psi = 11.01; thetai=0.3; thetas=0.4; dtheta = 0.247; dt = 0.166
# i = [1.08, 1.26, 1.56, 1.92, 2.22, 2.58, 3.84, 6.84, 19.08, \
#      9.90, 4.86, 3.12, 2.52, 2.16, 1.68, 1.44, 1.14, 1.02]
# # K, dt, theta_i, theta_s, psi, i
# a = GreenAmpt(K, dt, thetai, thetas, psi, i)
# F, f, t =a.F_f()
# #plt.plot(t, F)
# plt.plot(t[1:], f[1:])