#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ПКВ-оптимизатор с гидростатикой (Tkinter + Matplotlib, bmh)
- Первая вкладка: Q_day(T_acc) и f_on(T_acc) + блоки оптимумов (по Q и по f_on) и «перегиб f_on».
- Вторая вкладка: p_wb(t), q_in(t), q_pump(t) на 4 полных цикла (верх — энерг.оптимум, низ — Q-оптимум).
- [FIX] Динамика в «стопе»: насос=0, приток=a, p_wb=p_wb0 (осмысленный вид графиков).
- Дефолтные параметры — как на скрине пользователя.
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use('bmh')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.scrolledtext import ScrolledText

# -------- Константы --------
G = 9.80665      # м/с^2
P_ATM = 101325.0 # Па в 1 атм

# -------- Геометрия/гидростатика --------
def ring_area(d_obs_m: float, d_nkt_m: float) -> float:
    return math.pi / 4.0 * (d_obs_m**2 - d_nkt_m**2)

def gamma_atma_per_m(rho_kgm3: float) -> float:
    return (rho_kgm3 * G) / P_ATM   # атм/м

# -------- Накопление --------
def accumulation_volume(T_accum_sec: float, k_prod_m3dayatma: float,
                        p_res_atma: float, p_wb0_atma: float,
                        rho_kgm3: float, S_m2: float):
    k_sec = k_prod_m3dayatma / 86400.0
    Delta_p0 = p_res_atma - p_wb0_atma
    gamma = gamma_atma_per_m(rho_kgm3)
    a = k_sec * Delta_p0            # м3/с
    b = k_sec * gamma / S_m2        # 1/с
    if b <= 1e-18:
        x0 = a * T_accum_sec
    else:
        x0 = (a / b) * (1.0 - math.exp(-b * T_accum_sec))
    return x0, a, b

# -------- Откачка --------
def x_in_ramp(t, x0, a, b, q_opt, tau):
    eb = math.exp(-b * t)
    return (
        x0 * eb
        + (a / b) * (1.0 - eb)
        - (q_opt / tau) * (t / b - 1.0 / b**2)
        - (q_opt / tau) * (eb / b**2)
    )

def pump_time_with_hydro(x0, a, b, q_opt, tau):
    f0   = x_in_ramp(0.0,  x0, a, b, q_opt, tau)
    f_tau= x_in_ramp(tau,  x0, a, b, q_opt, tau)

    if f0 <= 0:
        return 0.0, "ramp"
    if f_tau <= 0:
        lo, hi = 0.0, tau
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = x_in_ramp(mid, x0, a, b, q_opt, tau)
            if fm <= 0: hi = mid
            else:       lo = mid
            if hi - lo < 1e-7: break
        return 0.5 * (lo + hi), "ramp"

    if q_opt <= a:
        return float('inf'), "impossible"
    x_tau = f_tau
    T_pump_sec = tau + (1.0 / b) * math.log(1.0 + (b * x_tau) / (q_opt - a))
    return T_pump_sec, "plate"

def pumped_volume_through_pump(T_pump_sec, q_opt, tau, regime):
    if not math.isfinite(T_pump_sec) or T_pump_sec <= 0: return 0.0
    if regime == "ramp":
        return (q_opt / (2.0 * tau)) * T_pump_sec**2
    else:
        return q_opt * (T_pump_sec - 0.5 * tau)

# -------- Метрики цикла --------
def metrics_for_Taccum_hydro(T_accum_min: float,
                             k_prod_m3dayatma: float,
                             q_esp_m3day_opt: float,
                             p_wb0_atma: float,
                             p_res_atma: float,
                             rho_kgm3: float,
                             d_obs_m: float,
                             d_nkt_m: float,
                             tau_ramp_sec: float,
                             add_stop_equals_tau: bool):
    S = ring_area(d_obs_m, d_nkt_m)
    q_opt = q_esp_m3day_opt / 86400.0
    T_accum_sec = T_accum_min * 60.0

    x0, a, b = accumulation_volume(T_accum_sec, k_prod_m3dayatma, p_res_atma, p_wb0_atma, rho_kgm3, S)
    T_pump_sec, regime = pump_time_with_hydro(x0, a, b, q_opt, tau_ramp_sec)

    T_stop_sec  = tau_ramp_sec if add_stop_equals_tau else 0.0
    T_total_sec = T_accum_sec + T_pump_sec + T_stop_sec
    T_total_min = T_total_sec / 60.0

    if not math.isfinite(T_total_min) or T_total_min <= 0:
        return dict(
            q_opt=q_opt, S=S, a=a, b=b, x0=x0, regime=regime,
            T_pump_min=float('inf'), T_total_min=float('inf'),
            N_cycles_day=0.0, Q_day_m3=0.0, V_pump_m3=0.0, f_on=1.0
        )

    V_pump = pumped_volume_through_pump(T_pump_sec, q_opt, tau_ramp_sec, regime)
    N      = 86400.0 / T_total_sec
    Q_day  = V_pump * N
    f_on   = (T_pump_sec + T_stop_sec) / T_total_sec

    return dict(
        q_opt=q_opt, S=S, a=a, b=b, x0=x0, regime=regime,
        T_pump_min=T_pump_sec / 60.0, T_total_min=T_total_min,
        N_cycles_day=N, Q_day_m3=Q_day, V_pump_m3=V_pump, f_on=f_on
    )

# -------- Оптимизации --------
def optimize_Taccum_hydro(Tmin: float, Tmax: float,
                          k_prod_m3dayatma: float,
                          q_esp_m3day_opt: float,
                          p_wb0_atma: float,
                          p_res_atma: float,
                          rho_kgm3: float,
                          d_obs_m: float,
                          d_nkt_m: float,
                          tau_ramp_sec: float,
                          add_stop_equals_tau: bool):
    T_grid = np.linspace(Tmin, Tmax, 1000)
    Q_grid, Fon_grid, M_grid = [], [], []
    for T in T_grid:
        m = metrics_for_Taccum_hydro(T, k_prod_m3dayatma, q_esp_m3day_opt, p_wb0_atma,
                                     p_res_atma, rho_kgm3, d_obs_m, d_nkt_m,
                                     tau_ramp_sec, add_stop_equals_tau)
        Q_grid.append(m['Q_day_m3'])
        Fon_grid.append(m['f_on'])
        M_grid.append(m)
    Q_grid   = np.array(Q_grid)
    Fon_grid = np.array(Fon_grid)

    idx_q = int(np.nanargmax(Q_grid))
    idx_e = int(np.nanargmin(Fon_grid))

    return dict(
        T_grid=T_grid, Q_grid=Q_grid, Fon_grid=Fon_grid,
        T_best_q=float(T_grid[idx_q]), best_q=M_grid[idx_q],
        T_best_e=float(T_grid[idx_e]), best_e=M_grid[idx_e],
        metrics_list=M_grid
    )

# -------- «Перегиб» f_on(T) --------
def find_knee_point(T, F):
    T = np.asarray(T); F = np.asarray(F)
    F_line = F[0] + (F[-1] - F[0]) * (T - T[0]) / (T[-1] - T[0])
    diff = F_line - F
    idx = int(np.argmax(diff))
    return idx

# -------- [FIX] Синтез временных рядов 1 цикла --------
def simulate_one_cycle(T_acc_min,
                       k_prod_m3dayatma, q_esp_m3day_opt,
                       p_wb0_atma, p_res_atma,
                       rho_kgm3, d_obs_m, d_nkt_m,
                       tau_ramp_sec, add_stop_equals_tau,
                       dt_sec=1.0):
    """
    В накоплении: x(t) по аналитике; p_wb = p_wb0 + gamma*x/S; q_in = a - b*x; q_pump = 0.
    В откачке:    x(t) по аналитике (разгон/полка); q_pump = q_opt*t/tau → q_opt; q_in = a - b*x.
    [FIX] Стоп:   насос=0, p_wb = p_wb0 (уровень ~0), НО приток идёт при этом dp=Δp0 ⇒ q_in = a.
                  Это убирает «сломанные» ступени и даёт физически осмысленный график.
    """
    S = ring_area(d_obs_m, d_nkt_m)
    gamma = gamma_atma_per_m(rho_kgm3)
    k_sec = k_prod_m3dayatma/86400.0
    a = k_sec*(p_res_atma - p_wb0_atma)
    b = (k_sec*gamma)/S
    q_opt = q_esp_m3day_opt/86400.0

    # --- накопление
    T_acc_sec = 60.0*T_acc_min
    t_acc = np.arange(0.0, T_acc_sec, dt_sec)
    if t_acc.size == 0 or t_acc[-1] < T_acc_sec: t_acc = np.append(t_acc, T_acc_sec)
    if b <= 1e-18:
        x_acc = a*t_acc
    else:
        x_acc = (a/b)*(1.0 - np.exp(-b*t_acc))
    p_acc = p_wb0_atma + gamma*(x_acc/S)
    qin_acc = a - b*x_acc
    qp_acc = np.zeros_like(t_acc)

    x0 = x_acc[-1]  # старт откачки

    # --- откачка
    T_pump_sec, regime = pump_time_with_hydro(x0, a, b, q_opt, tau_ramp_sec)
    t_pump = np.arange(0.0, T_pump_sec, dt_sec)
    if t_pump.size == 0 or t_pump[-1] < T_pump_sec: t_pump = np.append(t_pump, T_pump_sec)

    x_pump = np.empty_like(t_pump)
    qp_pump = np.empty_like(t_pump)
    for i, t in enumerate(t_pump):
        if t <= tau_ramp_sec:
            x_val = x_in_ramp(t, x0, a, b, q_opt, tau_ramp_sec)
            qp = q_opt*(t/tau_ramp_sec)
        else:
            x_tau = x_in_ramp(tau_ramp_sec, x0, a, b, q_opt, tau_ramp_sec)
            dt = t - tau_ramp_sec
            x_val = x_tau*np.exp(-b*dt) + (a - q_opt)/b*(1.0 - np.exp(-b*dt))
            qp = q_opt
        x_pump[i] = max(x_val, 0.0)
        qp_pump[i] = qp

    p_pump = p_wb0_atma + gamma*(x_pump/S)
    qin_pump = a - b*x_pump

    # --- стоп (накладка по времени)
    if add_stop_equals_tau:
        T_stop_sec = tau_ramp_sec
        t_stop = np.arange(0.0, T_stop_sec, dt_sec)
        if t_stop.size == 0 or t_stop[-1] < T_stop_sec: t_stop = np.append(t_stop, T_stop_sec)
        p_stop = np.full_like(t_stop, p_wb0_atma)           # уровень ~0 ⇒ p_wb ~ p_wb0
        qin_stop = np.full_like(t_stop, a)                  # [FIX] приток идёт при dp=Δp0
        qp_stop = np.zeros_like(t_stop)                     # насос выключен

        # склейка
        t = np.concatenate([t_acc,
                            T_acc_sec + t_pump,
                            T_acc_sec + T_pump_sec + t_stop])
        p = np.concatenate([p_acc, p_pump, p_stop])
        qin = np.concatenate([qin_acc, qin_pump, qin_stop])
        qp = np.concatenate([qp_acc, qp_pump, qp_stop])
    else:
        t = np.concatenate([t_acc, T_acc_sec + t_pump])
        p = np.concatenate([p_acc, p_pump])
        qin = np.concatenate([qin_acc, qin_pump])
        qp = np.concatenate([qp_acc, qp_pump])

    return dict(t_sec=t, p_wb_atm=p, q_in_m3s=qin, q_pump_m3s=qp,
                T_cycle_sec=t[-1])

def simulate_four_cycles(T_acc_min, params, dt_sec=1.0):
    one = simulate_one_cycle(T_acc_min, **params, dt_sec=dt_sec)
    t0, p0, qi0, qp0 = one['t_sec'], one['p_wb_atm'], one['q_in_m3s'], one['q_pump_m3s']
    Tcyc = one['T_cycle_sec']
    t = np.array([]); p = np.array([]); qi = np.array([]); qp = np.array([])
    for k in range(4):
        shift = k*Tcyc
        t = np.concatenate([t, t0 + shift])
        p = np.concatenate([p, p0])
        qi = np.concatenate([qi, qi0])
        qp = np.concatenate([qp, qp0])
    return dict(t_sec=t, p_wb_atm=p, q_in_m3s=qi, q_pump_m3s=qp)

# -------- GUI --------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ПКВ оптимизация (гидростатика, плотность/диаметры) — bmh")
        self.geometry("1340x880")
        self._build()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        # ---- вкладка "Расчёт"
        tab_calc = ttk.Frame(nb)
        nb.add(tab_calc, text="Расчёт")

        left = ttk.Frame(tab_calc, padding=(12, 10))
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(tab_calc, padding=(8, 10))
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        def add_field(row, label, var, width=12):
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2)
            e = ttk.Entry(left, textvariable=var, width=width)
            e.grid(row=row, column=1, sticky="w", pady=2)
            return e

        # --- ДЕФОЛТЫ (как на твоём втором скрине)
        self.var_k      = tk.StringVar(value="1.35")   # м3/сут/атм
        self.var_qesp   = tk.StringVar(value="124")    # м3/сут
        self.var_pwb0   = tk.StringVar(value="43")     # атм
        self.var_pres   = tk.StringVar(value="108")    # атм
        self.var_rho    = tk.StringVar(value="820")    # кг/м3
        self.var_dobs   = tk.StringVar(value="0.159")  # м
        self.var_dnkt   = tk.StringVar(value="0.062")  # м
        self.var_tau    = tk.StringVar(value="25")     # с
        self.var_tmin   = tk.StringVar(value="2")      # мин
        self.var_tmax   = tk.StringVar(value="25")     # мин
        self.var_stop_eq_tau = tk.BooleanVar(value=True)

        add_field(0, "k_prod, м³/сут/атм:", self.var_k)
        add_field(1, "q_esp_opt, м³/сут:",  self.var_qesp)
        add_field(2, "p_wb0, атм:",         self.var_pwb0)
        add_field(3, "p_res, атм:",         self.var_pres)
        add_field(4, "ρ (плотность), кг/м³:", self.var_rho)
        add_field(5, "d_obs, м:",           self.var_dobs)
        add_field(6, "d_nkt, м:",           self.var_dnkt)
        add_field(7, "τ (разгон), с:",      self.var_tau)
        ttk.Checkbutton(left, text="Учитывать останов (= τ)", variable=self.var_stop_eq_tau)\
            .grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Separator(left).grid(row=9, column=0, columnspan=2, sticky="ew", pady=6)
        add_field(10, "T_accum min, мин:",  self.var_tmin)
        add_field(11, "T_accum max, мин:",  self.var_tmax)

        ttk.Button(left, text="Рассчитать", command=self.run_calc)\
            .grid(row=12, column=0, columnspan=2, sticky="we", pady=(10, 4))

        # Метрики
        metrics = ttk.LabelFrame(right, text="Результаты (Q-оптимум и min f_on)")
        metrics.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 6))

        self.metric_vars = {
            "q_opt": tk.StringVar(), "S": tk.StringVar(), "gamma": tk.StringVar(),
            "a": tk.StringVar(), "b": tk.StringVar(),
            "T_accum_q": tk.StringVar(), "T_pump_q": tk.StringVar(), "N_q": tk.StringVar(),
            "Q_day_q": tk.StringVar(), "V_pump_q": tk.StringVar(), "f_on_q": tk.StringVar(), "regime_q": tk.StringVar(),
            "T_accum_e": tk.StringVar(), "T_pump_e": tk.StringVar(), "N_e": tk.StringVar(),
            "Q_day_e": tk.StringVar(), "V_pump_e": tk.StringVar(), "f_on_e": tk.StringVar(), "regime_e": tk.StringVar(),
            "T_accum_k": tk.StringVar(), "T_pump_k": tk.StringVar(), "N_k": tk.StringVar(),
            "Q_day_k": tk.StringVar(), "V_pump_k": tk.StringVar(), "f_on_k": tk.StringVar(), "regime_k": tk.StringVar(),
            "warn": tk.StringVar()
        }

        row = 0
        def add_metric(label, key, col):
            ttk.Label(metrics, text=label).grid(row=row, column=col, sticky="w", padx=(8, 4), pady=2)
            ttk.Label(metrics, textvariable=self.metric_vars[key]).grid(row=row, column=col+1, sticky="w", pady=2)

        add_metric("q_opt:", "q_opt", 0)
        add_metric("S (м²):", "S", 2)
        add_metric("γ (атм/м):", "gamma", 4)
        row += 1
        add_metric("a (м³/с):", "a", 0)
        add_metric("b (1/с):", "b", 2)
        ttk.Label(metrics, textvariable=self.metric_vars["warn"], foreground="#b33").grid(
            row=row, column=4, columnspan=2, sticky="w", padx=8, pady=(2, 2)
        )
        row += 1
        add_metric("Q-опт. T_accum*, мин:", "T_accum_q", 0)
        add_metric("T_pump*, мин:",         "T_pump_q",  2)
        add_metric("f_on*:",                "f_on_q",    4)
        row += 1
        add_metric("N*, 1/сут:",            "N_q",       0)
        add_metric("V_pump*, м³:",          "V_pump_q",  2)
        add_metric("Q_day*, м³/сут:",       "Q_day_q",   4)
        row += 1
        add_metric("режим:",                "regime_q",  0)
        ttk.Separator(metrics).grid(row=row, column=0, columnspan=6, sticky="ew", pady=(6, 2))
        row += 1
        add_metric("Энерг.опт. T_accum°, мин:", "T_accum_e", 0)
        add_metric("T_pump°, мин:",              "T_pump_e",  2)
        add_metric("f_on° (min):",               "f_on_e",    4)
        row += 1
        add_metric("N°, 1/сут:",                 "N_e",       0)
        add_metric("V_pump°, м³:",               "V_pump_e",  2)
        add_metric("Q_day°, м³/сут:",            "Q_day_e",   4)
        row += 1
        add_metric("режим:",                     "regime_e",  0)
        ttk.Separator(metrics).grid(row=row, column=0, columnspan=6, sticky="ew", pady=(6, 2))
        row += 1
        add_metric("Перегиб T_accum^, мин:", "T_accum_k", 0)
        add_metric("T_pump^, мин:",          "T_pump_k",  2)
        add_metric("f_on^:",                 "f_on_k",    4)
        row += 1
        add_metric("N^, 1/сут:",             "N_k",       0)
        add_metric("V_pump^, м³:",           "V_pump_k",  2)
        add_metric("Q_day^, м³/сут:",        "Q_day_k",   4)
        row += 1
        add_metric("режим:",                 "regime_k",  0)

        # ---- два графика в строку
        plots_row = ttk.Frame(right)
        plots_row.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        figL = plt.figure(figsize=(6.6, 4.8)); self.axQ = figL.add_subplot(111)
        self.canvasQ = FigureCanvasTkAgg(figL, master=plots_row)
        self.canvasQ.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        figR = plt.figure(figsize=(12.6, 4.8)); self.axF = figR.add_subplot(111)
        self.canvasF = FigureCanvasTkAgg(figR, master=plots_row)
        self.canvasF.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # ---- вкладка "Динамика за 4 цикла"
        tab_dyn = ttk.Frame(nb); nb.add(tab_dyn, text="Динамика за 4 цикла")
        dyn_frame = ttk.Frame(tab_dyn, padding=(8, 10)); dyn_frame.pack(fill=tk.BOTH, expand=True)

        figD = plt.figure(figsize=(10.5, 7.6))
        self.axD_top = figD.add_subplot(2, 1, 1)
        self.axD_bot = figD.add_subplot(2, 1, 2)
        self.canvasD = FigureCanvasTkAgg(figD, master=dyn_frame)
        self.canvasD.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---- вкладка "Модель"
        tab_model = ttk.Frame(nb); nb.add(tab_model, text="Модель")
        txt = ScrolledText(tab_model, wrap="word", height=38)
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        txt.insert("1.0",
            "Формулы (гидростатика): ẋ=a-bx; ẋ=a-bx-q_p(t); T_pump (ramp/plate); "
            "V_pump; Q_day; f_on; a=k_secΔp0; b=k_secγ/S; γ=ρg/p_atm; S=π/4(d_obs²-d_nkt²) "
        )
        txt.configure(state="disabled")

        self.run_calc()

    def run_calc(self):
        try:
            k      = float(self.var_k.get())
            qesp   = float(self.var_qesp.get())
            pwb0   = float(self.var_pwb0.get())
            pres   = float(self.var_pres.get())
            rho    = float(self.var_rho.get())
            dobs   = float(self.var_dobs.get())
            dnkt   = float(self.var_dnkt.get())
            tau    = float(self.var_tau.get())
            tmin   = float(self.var_tmin.get())
            tmax   = float(self.var_tmax.get())
            add_stop = bool(self.var_stop_eq_tau.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте ввод чисел."); return

        if tmin <= 0 or tmax <= 0 or tmax < tmin:
            messagebox.showerror("Ошибка", "Неверные границы T_accum."); return
        if pres <= pwb0:
            messagebox.showerror("Ошибка", "p_res должно быть > p_wb0."); return
        if tau <= 0:
            messagebox.showerror("Ошибка", "τ должно быть > 0."); return
        if dobs <= dnkt:
            messagebox.showerror("Ошибка", "d_obs должно быть > d_nkt."); return
        if rho <= 0:
            messagebox.showerror("Ошибка", "ρ должно быть > 0."); return

        opt = optimize_Taccum_hydro(tmin, tmax, k, qesp, pwb0, pres, rho, dobs, dnkt, tau, add_stop)
        T_grid, Q_grid, Fon_grid = opt['T_grid'], opt['Q_grid'], opt['Fon_grid']
        T_best_q, best_q = opt['T_best_q'], opt['best_q']
        T_best_e, best_e = opt['T_best_e'], opt['best_e']

        idx_knee = find_knee_point(T_grid, Fon_grid)
        T_knee = float(T_grid[idx_knee])
        m_knee = metrics_for_Taccum_hydro(T_knee, k, qesp, pwb0, pres, rho, dobs, dnkt, tau, add_stop)

        # служебные
        S = ring_area(dobs, dnkt); gamma = gamma_atma_per_m(rho)
        self.metric_vars["q_opt"].set(f"{best_q['q_opt']*86400:.3f} м³/сут")
        self.metric_vars["S"].set(f"{S:.6f}"); self.metric_vars["gamma"].set(f"{gamma:.6f}")
        self.metric_vars["a"].set(f"{best_q['a']:.6e}"); self.metric_vars["b"].set(f"{best_q['b']:.6e}")
        warn = ""
        if best_q['regime']=="impossible" or best_e['regime']=="impossible":
            warn = "Опорожнение невозможно (q_opt ≤ a). Повышайте q_esp_opt или снижайте Δp0/ρ/S."
        self.metric_vars["warn"].set(warn)

        # метрики
        for (pref, Tacc, m) in (("q", T_best_q, best_q), ("e", T_best_e, best_e), ("k", T_knee, m_knee)):
            self.metric_vars[f"T_accum_{pref}"].set(f"{Tacc:.3f}")
            self.metric_vars[f"T_pump_{pref}"].set(f"{m['T_pump_min']:.3f}")
            self.metric_vars[f"N_{pref if pref!='q' else 'q'}"].set(f"{m['N_cycles_day']:.3f}")
            self.metric_vars[f"V_pump_{pref}"].set(f"{m['V_pump_m3']:.4f}")
            self.metric_vars[f"Q_day_{pref}"].set(f"{m['Q_day_m3']:.3f}")
            self.metric_vars[f"f_on_{pref}"].set(f"{m['f_on']:.4f}")
            self.metric_vars[f"regime_{pref}"].set(m['regime'])

        # --- ЛЕВЫЙ график: Q_day(T)
        self.axQ.clear()
        self.axQ.plot(T_grid, Q_grid)
        self.axQ.scatter([T_best_q], [best_q['Q_day_m3']], label="Оптимум по накопленной добыче")
        self.axQ.axvline(T_best_e, linestyle="--", linewidth=1.2)
        self.axQ.scatter([T_knee], [np.interp(T_knee, T_grid, Q_grid)], color="green", zorder=4, label="Оптимум по времени работы УЭЦН в цикле")
        self.axQ.set_title("Добыча от длительности цикла накопления")
        self.axQ.set_xlabel("Время накопления, мин"); self.axQ.set_ylabel("Q_day, м³/сут")
        self.axQ.legend(loc="best", fontsize=9, framealpha=0.85); self.axQ.grid(True)
        self.axQ.figure.tight_layout(); self.canvasQ.draw_idle()

        # --- ПРАВЫЙ график: f_on(T)
        self.axF.clear()
        self.axF.plot(T_grid, Fon_grid, label="f_on от времени накопления")
        self.axF.scatter([T_best_e], [best_e['f_on']], label="min f_on", zorder=3)
        self.axF.scatter([T_knee], [Fon_grid[idx_knee]], color="green", zorder=4, label="перегиб f_on")
        self.axF.set_title("Откачка за цикл (f_on)") 
        self.axF.set_xlabel("Время накопления, мин"); self.axF.set_ylabel("Откачка за весь цикл, д.ед.")
        self.axF.legend(loc="best", fontsize=9, framealpha=0.85); self.axF.grid(True)
        self.axF.figure.tight_layout(); self.canvasF.draw_idle()

        # ----------- Вкладка "Динамика за 4 цикла" -----------
        sim_params = dict(
            k_prod_m3dayatma=k, q_esp_m3day_opt=qesp,
            p_wb0_atma=pwb0, p_res_atma=pres,
            rho_kgm3=rho, d_obs_m=dobs, d_nkt_m=dnkt,
            tau_ramp_sec=tau, add_stop_equals_tau=add_stop
        )
        dyn_e = simulate_four_cycles(T_best_e, sim_params, dt_sec=1.0)
        dyn_q = simulate_four_cycles(T_best_q, sim_params, dt_sec=1.0)

        # ===== Верх: Энерго-оптимум =====
        self.axD_top.clear()
        t_e_min = dyn_e['t_sec'] / 60.0
        p_e      = dyn_e['p_wb_atm']
        qi_e_day = dyn_e['q_in_m3s']   * 86400.0  # м³/сут
        qp_e_day = dyn_e['q_pump_m3s'] * 86400.0  # м³/сут

        ax = self.axD_top
        ln_p, = ax.plot(t_e_min, p_e, color='tab:blue', label="p_wb, атм", linewidth=1.6)
        ax.set_ylabel("p_wb, атм")
        ax.set_title("Динамика за 4 цикла — Энерго-оптимум (min f_on)")
        ax.grid(True)

        axr = ax.twinx()
        ln_qi, = axr.plot(t_e_min, qi_e_day, color='tab:orange', label="q_in, м³/сут", linewidth=1.4)
        ln_qp, = axr.plot(t_e_min, qp_e_day, color='tab:red',    label="q_pump, м³/сут", linewidth=1.4)
        axr.set_ylabel("дебиты, м³/сут")

        lines = [ln_p, ln_qi, ln_qp]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=8)

        # ===== Низ: Оптимум по накопленной =====
        self.axD_bot.clear()
        t_q_min = dyn_q['t_sec'] / 60.0
        p_q      = dyn_q['p_wb_atm']
        qi_q_day = dyn_q['q_in_m3s']   * 86400.0  # м³/сут
        qp_q_day = dyn_q['q_pump_m3s'] * 86400.0  # м³/сут

        ax = self.axD_bot
        ln_p2, = ax.plot(t_q_min, p_q, color='tab:blue', label="p_wb, атм", linewidth=1.6)
        ax.set_xlabel("Время, мин")
        ax.set_ylabel("p_wb, атм")
        ax.set_title("Динамика за 4 цикла — Оптимум по накопленной (max Q_day)")
        ax.grid(True)

        axr = ax.twinx()
        ln_qi2, = axr.plot(t_q_min, qi_q_day, color='tab:orange', label="q_in, м³/сут", linewidth=1.4)
        ln_qp2, = axr.plot(t_q_min, qp_q_day, color='tab:red',    label="q_pump, м³/сут", linewidth=1.4)
        axr.set_ylabel("дебиты, м³/сут")

        lines2 = [ln_p2, ln_qi2, ln_qp2]
        labels2 = [l.get_label() for l in lines2]
        ax.legend(lines2, labels2, loc="upper right", fontsize=8)

        self.canvasD.draw_idle()


if __name__ == "__main__":
    App().mainloop()
