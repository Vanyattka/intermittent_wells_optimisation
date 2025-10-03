#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ПКВ-оптимизатор с гидростатикой (Tkinter + Matplotlib, bmh)
Обновления:
- Отдельный график f_on(T_accum)
- Два графика в одну строку (слева Q_day, справа f_on)
- Поиск «точки перегиба» (knee) на f_on(T) и вывод её параметров
- Зелёные метки точки перегиба на обоих графиках
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
    # gamma = rho*g / (Pa_per_atm)  [атм/м]
    return (rho_kgm3 * G) / P_ATM

# -------- Накопление --------
def accumulation_volume(T_accum_sec: float, k_prod_m3dayatma: float,
                        p_res_atma: float, p_wb0_atma: float,
                        rho_kgm3: float, S_m2: float):
    """
    x0 (= dv) — объём, накопленный за фазу накопления (м^3)
    a, b      — коэффициенты ОДУ (м^3/с и 1/с)
    """
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
    """Решение x(t) на разгоне (0..tau) для ẋ = a - b x - (q_opt/τ) t."""
    eb = math.exp(-b * t)
    return (
        x0 * eb
        + (a / b) * (1.0 - eb)
        - (q_opt / tau) * (t / b - 1.0 / b**2)
        - (q_opt / tau) * (eb / b**2)
    )

def pump_time_with_hydro(x0, a, b, q_opt, tau):
    """
    Возвращает (T_pump_sec, regime)
      regime: "ramp"  — опустошили на разгоне
              "plate" — дошли до полки и опустошили на полке
              "impossible" — q_opt <= a, опустошить нельзя
    """
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

    # не успели на разгоне → полка
    if q_opt <= a:
        return float('inf'), "impossible"
    x_tau = f_tau
    T_pump_sec = tau + (1.0 / b) * math.log(1.0 + (b * x_tau) / (q_opt - a))
    return T_pump_sec, "plate"

def pumped_volume_through_pump(T_pump_sec, q_opt, tau, regime):
    """Интеграл q_pump dt (без времени остановки)."""
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
        return dict(q_opt=q_opt, S=S, a=a, b=b, x0=x0, regime=regime,
                    T_pump_min=float('inf'), T_total_min=float('inf'),
                    N_cycles_day=0.0, Q_day_m3=0.0, V_pump_m3=0.0, f_on=1.0)

    V_pump = pumped_volume_through_pump(T_pump_sec, q_opt, tau_ramp_sec, regime)
    N      = 86400.0 / T_total_sec
    Q_day  = V_pump * N
    f_on   = (T_pump_sec + T_stop_sec) / T_total_sec

    return dict(q_opt=q_opt, S=S, a=a, b=b, x0=x0, regime=regime,
                T_pump_min=T_pump_sec / 60.0, T_total_min=T_total_min,
                N_cycles_day=N, Q_day_m3=Q_day, V_pump_m3=V_pump, f_on=f_on)

# -------- Оптимизации по Q и по f_on --------
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
    T_grid = np.linspace(Tmin, Tmax, 800)
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

# -------- Точка перегиба (knee) на f_on(T) --------
def find_knee_point(T, F):
    """
    Поиск «локтя» (Kneedle): максимальная вертикальная разность
    между кривой F(T) и прямой, соединяющей концы (T0,F0)-(T1,F1).
    f_on убывает и выполаживается вправо — локоть получается в месте перегиба.
    """
    T = np.asarray(T); F = np.asarray(F)
    # прямая по концам:
    F_line = F[0] + (F[-1] - F[0]) * (T - T[0]) / (T[-1] - T[0])
    # для убывающей кривой ищем максимум (F_line - F)
    diff = F_line - F
    idx = int(np.argmax(diff))
    return idx

# -------- GUI --------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ПКВ оптимизация (гидростатика, плотность/диаметры) — bmh")
        self.geometry("1320x820")
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

        # Ввод
        self.var_k      = tk.StringVar(value="0.05")   # м3/сут/атм
        self.var_qesp   = tk.StringVar(value="45")     # м3/сут
        self.var_pwb0   = tk.StringVar(value="48")     # атм
        self.var_pres   = tk.StringVar(value="195")    # атм
        self.var_rho    = tk.StringVar(value="800")    # кг/м3
        self.var_dobs   = tk.StringVar(value="0.159")  # м
        self.var_dnkt   = tk.StringVar(value="0.073")  # м
        self.var_tau    = tk.StringVar(value="15")     # с
        self.var_tmin   = tk.StringVar(value="2")      # мин
        self.var_tmax   = tk.StringVar(value="25")     # мин
        self.var_stop_eq_tau = tk.BooleanVar(value=False)

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

        # Метрики (Q-оптимум и энерго-оптимум)
        metrics = ttk.LabelFrame(right, text="Результаты (Q-оптимум и min f_on)")
        metrics.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 6))

        self.metric_vars = {
            "q_opt": tk.StringVar(), "S": tk.StringVar(), "gamma": tk.StringVar(),
            "a": tk.StringVar(), "b": tk.StringVar(),
            "T_accum_q": tk.StringVar(), "T_pump_q": tk.StringVar(), "N_q": tk.StringVar(),
            "Q_day_q": tk.StringVar(), "V_pump_q": tk.StringVar(), "f_on_q": tk.StringVar(), "regime_q": tk.StringVar(),
            "T_accum_e": tk.StringVar(), "T_pump_e": tk.StringVar(), "N_e": tk.StringVar(),
            "Q_day_e": tk.StringVar(), "V_pump_e": tk.StringVar(), "f_on_e": tk.StringVar(), "regime_e": tk.StringVar(),
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

        # Блок «Оптимум (перегиб f_on)»
        self.knee_frame = ttk.LabelFrame(right, text="Оптимум (перегиб f_on)")
        self.knee_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 8))
        self.knee_vars = {
            "T_accum": tk.StringVar(), "T_pump": tk.StringVar(), "f_on": tk.StringVar(),
            "N": tk.StringVar(), "V_pump": tk.StringVar(), "Q_day": tk.StringVar(), "regime": tk.StringVar()
        }
        r=0
        def add_knee(label, key, col):
            ttk.Label(self.knee_frame, text=label).grid(row=r, column=col, sticky="w", padx=(8,4), pady=2)
            ttk.Label(self.knee_frame, textvariable=self.knee_vars[key]).grid(row=r, column=col+1, sticky="w", pady=2)
        add_knee("T_accum^ (мин):", "T_accum", 0)
        add_knee("T_pump^ (мин):",  "T_pump",  2)
        add_knee("f_on^:",          "f_on",    4); r+=1
        add_knee("N^ (1/сут):",     "N",       0)
        add_knee("V_pump^ (м³):",   "V_pump",  2)
        add_knee("Q_day^ (м³/сут):","Q_day",   4); r+=1
        add_knee("режим:",          "regime",  0)

        # ---- два графика в строку
        plots_row = ttk.Frame(right)
        plots_row.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # левый: Q_day(T)
        figL = plt.figure(figsize=(6.4, 4.8))
        self.axQ = figL.add_subplot(111)
        self.canvasQ = FigureCanvasTkAgg(figL, master=plots_row)
        self.canvasQ.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # правый: f_on(T)
        figR = plt.figure(figsize=(6.4, 4.8))
        self.axF = figR.add_subplot(111)
        self.canvasF = FigureCanvasTkAgg(figR, master=plots_row)
        self.canvasF.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # ---- вкладка "Модель"
        tab_model = ttk.Frame(nb)
        nb.add(tab_model, text="Модель")
        txt = ScrolledText(tab_model, wrap="word", height=40)
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        txt.insert("1.0", (
            "Формулы (гидростатика):\n"
            "Накопление: ẋ = a - b x, x(0)=0 → x(t)=(a/b)(1-e^{-bt})\n"
            "Откачка: ẋ = a - b x - q_p(t), q_p(t)=q_opt·t/τ на [0,τ], далее q_opt\n"
            "T_pump: ramp → корень x(t)=0; plate → τ + (1/b)·ln(1 + b·x_τ/(q_opt - a)), q_opt>a\n"
            "V_pump: ramp→ q_opt t*²/(2τ); plate→ q_opt (T_pump - τ/2)\n"
            "Q_day = V_pump · 86400/(T_accum + T_pump + T_stop)\n"
            "f_on  = (T_pump+T_stop)/(T_accum+T_pump+T_stop)\n"
            "a = k_sec·(p_res - p_wb0),  b = k_sec·γ/S,  γ=ρg/p_atm,  S=π/4(d_obs^2-d_nkt^2)\n"
        ))
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
            messagebox.showerror("Ошибка", "Проверьте, что все поля заполнены числами.")
            return

        if tmin <= 0 or tmax <= 0 or tmax < tmin:
            messagebox.showerror("Ошибка", "Неверные границы T_accum."); return
        if pres <= pwb0:
            messagebox.showerror("Ошибка", "p_res должно быть больше p_wb0."); return
        if tau <= 0:
            messagebox.showerror("Ошибка", "τ должно быть > 0."); return
        if dobs <= dnkt:
            messagebox.showerror("Ошибка", "d_obs должно быть > d_nkt."); return
        if rho <= 0:
            messagebox.showerror("Ошибка", "ρ должно быть > 0."); return

        # Оптимизация: максимум Q и минимум f_on
        opt = optimize_Taccum_hydro(tmin, tmax, k, qesp, pwb0, pres, rho, dobs, dnkt, tau, add_stop)
        T_grid, Q_grid, Fon_grid = opt['T_grid'], opt['Q_grid'], opt['Fon_grid']
        T_best_q, best_q = opt['T_best_q'], opt['best_q']
        T_best_e, best_e = opt['T_best_e'], opt['best_e']

        # Поиск «перегиба» f_on(T)
        idx_knee = find_knee_point(T_grid, Fon_grid)
        T_knee = float(T_grid[idx_knee])
        m_knee = metrics_for_Taccum_hydro(T_knee, k, qesp, pwb0, pres, rho, dobs, dnkt, tau, add_stop)

        # Служебные метрики
        S = ring_area(dobs, dnkt)
        gamma = gamma_atma_per_m(rho)
        self.metric_vars["q_opt"].set(f"{best_q['q_opt']*86400:.3f} м³/сут")
        self.metric_vars["S"].set(f"{S:.6f}")
        self.metric_vars["gamma"].set(f"{gamma:.6f}")
        self.metric_vars["a"].set(f"{best_q['a']:.6e}")
        self.metric_vars["b"].set(f"{best_q['b']:.6e}")
        warn = ""
        if best_q['regime'] == "impossible" or best_e['regime'] == "impossible":
            warn = "Опорожнение невозможно (q_opt ≤ a). Увеличьте q_esp_opt или снизьте Δp0/ρ/S."
        self.metric_vars["warn"].set(warn)

        # Заполнение метрик: Q-оптимум
        self.metric_vars["T_accum_q"].set(f"{T_best_q:.3f}")
        self.metric_vars["T_pump_q"].set(f"{best_q['T_pump_min']:.3f}")
        self.metric_vars["N_q"].set(f"{best_q['N_cycles_day']:.3f}")
        self.metric_vars["V_pump_q"].set(f"{best_q['V_pump_m3']:.4f}")
        self.metric_vars["Q_day_q"].set(f"{best_q['Q_day_m3']:.3f}")
        self.metric_vars["f_on_q"].set(f"{best_q['f_on']:.4f}")
        self.metric_vars["regime_q"].set(best_q['regime'])

        # Энерго-оптимум
        self.metric_vars["T_accum_e"].set(f"{T_best_e:.3f}")
        self.metric_vars["T_pump_e"].set(f"{best_e['T_pump_min']:.3f}")
        self.metric_vars["N_e"].set(f"{best_e['N_cycles_day']:.3f}")
        self.metric_vars["V_pump_e"].set(f"{best_e['V_pump_m3']:.4f}")
        self.metric_vars["Q_day_e"].set(f"{best_e['Q_day_m3']:.3f}")
        self.metric_vars["f_on_e"].set(f"{best_e['f_on']:.4f}")
        self.metric_vars["regime_e"].set(best_e['regime'])

        # Блок «Оптимум (перегиб f_on)»
        self.knee_vars["T_accum"].set(f"{T_knee:.3f}")
        self.knee_vars["T_pump"].set(f"{m_knee['T_pump_min']:.3f}")
        self.knee_vars["f_on"].set(f"{m_knee['f_on']:.4f}")
        self.knee_vars["N"].set(f"{m_knee['N_cycles_day']:.3f}")
        self.knee_vars["V_pump"].set(f"{m_knee['V_pump_m3']:.4f}")
        self.knee_vars["Q_day"].set(f"{m_knee['Q_day_m3']:.3f}")
        self.knee_vars["regime"].set(m_knee['regime'])

        # ---- ЛЕВЫЙ график: Q_day(T_accum)
        self.axQ.clear()
        self.axQ.plot(T_grid, Q_grid, label="Q_day(T_accum)")
        self.axQ.scatter([T_best_q], [best_q['Q_day_m3']], label="Q-оптимум")
        self.axQ.axvline(T_best_e, linestyle="--", linewidth=1.2, label=f"min f_on @ {T_best_e:.2f} мин")
        # зелёная точка — перегиб
        self.axQ.scatter([T_knee], [np.interp(T_knee, T_grid, Q_grid)], color="green", zorder=4, label="перегиб f_on")
        self.axQ.set_title("Q_day vs. T_accum (гидростатика)")
        self.axQ.set_xlabel("T_accum, мин")
        self.axQ.set_ylabel("Q_day, м³/сут")
        self.axQ.legend(loc="best", fontsize=9, framealpha=0.85)
        self.axQ.grid(True)
        self.axQ.figure.tight_layout()
        self.canvasQ.draw_idle()

        # === после того как посчитаны T_grid, Q_grid, Fon_grid, T_knee и построены оба графика ===

        if add_stop:
            # границы «оптимальной зоны»: от точки перегиба до правого края диапазона
            T_opt_lo = T_knee
            T_opt_hi = float(T_grid[-1])

            # левая ось: Q_day(T_accum)
            self.axQ.axvspan(T_opt_lo, T_opt_hi, alpha=0.18, color='green', label="оптимальная зона")
            yq_annot = float(np.interp(T_opt_lo, T_grid, Q_grid))
            self.axQ.annotate("оптимальная зона",
                            xy=(T_opt_lo, yq_annot), xytext=(8, 12),
                            textcoords="offset points", fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

            # правая ось: f_on(T_accum)
            self.axF.axvspan(T_opt_lo, T_opt_hi, alpha=0.18, color='green', label="оптимальная зона")
            yf_annot = float(np.interp(T_opt_lo, T_grid, Fon_grid))
            self.axF.annotate("оптимальная зона",
                            xy=(T_opt_lo, yf_annot), xytext=(8, 12),
                            textcoords="offset points", fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

            # обновляем легенды после добавления зоны
            self.axQ.legend(loc="best", fontsize=9, framealpha=0.85)
            self.axF.legend(loc="best", fontsize=9, framealpha=0.85)

        # перерисовка (если вставляешь до draw_idle — можно оставить как есть)
        self.canvasQ.draw_idle()
        self.canvasF.draw_idle()


        # ---- ПРАВЫЙ график: f_on(T_accum)
        self.axF.clear()
        self.axF.plot(T_grid, Fon_grid, label="f_on(T_accum)")
        self.axF.scatter([T_best_e], [best_e['f_on']], label="min f_on", zorder=3)
        # зелёная точка — перегиб
        self.axF.scatter([T_knee], [Fon_grid[idx_knee]], color="green", zorder=4, label="перегиб f_on")
        self.axF.set_title("Доля on-time f_on vs. T_accum")
        self.axF.set_xlabel("T_accum, мин")
        self.axF.set_ylabel("f_on, доля")
        self.axF.legend(loc="best", fontsize=9, framealpha=0.85)
        self.axF.grid(True)
        self.axF.figure.tight_layout()
        self.canvasF.draw_idle()

if __name__ == "__main__":
    App().mainloop()
