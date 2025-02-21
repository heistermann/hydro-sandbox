import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ipywidgets import IntSlider
from matplotlib.widgets import Button, Slider
from ipywidgets import IntText, link
import ipywidgets as widgets

def abc(P, a, b, c):
    """Das abc-Modell
    """
    assert(a+b<=1)
    assert(c<=1)
    df = pd.DataFrame(columns=["P","Q","Qd","Qb"], 
                      index=np.arange(len(P)), dtype="float")
    df["P"] = P
    # Fuer den Grundwassspeicher G muessen wir einen Startwert annehmen.
    G = 5.
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

def plot_abc_model(P):
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

# SCS-CN




cn2table = pd.read_csv("cn-table.csv", index_col="Landnutzung")
amctable = pd.read_csv("amc-table.csv", sep=";", index_col="cn")

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