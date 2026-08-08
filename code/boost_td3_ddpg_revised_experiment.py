from __future__ import annotations
import argparse, csv, os, random, itertools, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def write_csv(path, header, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def moving_average(x, w=9):
    x=np.asarray(x,float)
    if len(x)<w: return x
    k=np.ones(w)/w
    y=np.convolve(x,k,mode='same')
    # edge correction
    for i in range(w//2):
        y[i]=np.mean(x[:i+w//2+1]); y[-i-1]=np.mean(x[-(i+w//2+1):])
    return y

# -----------------------------
# Converter model and environment
# -----------------------------
@dataclass
class BoostParams:
    Vin0: float=100.0
    Vref: float=200.0
    L0: float=1e-3
    C0: float=470e-6
    R0: float=50.0
    Tc: float=5e-4        # controller update period = 500 us
    h: float=5e-5         # RK4 internal step = 50 us
    u_min: float=0.05
    u_max: float=0.90
    iL_clip: float=80.0
    vo_min: float=1.0
    vo_max: float=320.0

class BoostConverterEnv:
    def __init__(self,p:BoostParams,scenario='random',episode_steps=1000,seed=0):
        self.p=p; self.scenario=scenario; self.episode_steps=episode_steps
        self.rng=np.random.default_rng(seed)
        self._train_sched=None
        self.reset(scenario)

    def reset(self,scenario=None):
        if scenario is not None: self.scenario=scenario
        self.t=0
        self.prev_u=1.0-self.p.Vin0/self.p.Vref
        self.vo=self.p.Vref
        self.iL=self.p.Vref/self.p.R0/max(1e-9,(1.0-self.prev_u))
        if self.scenario=='random':
            self.vo=self.p.Vref+self.rng.uniform(-12,12)
            self.prev_u=float(np.clip(self.prev_u+self.rng.uniform(-0.035,0.035),self.p.u_min,self.p.u_max))
            self.iL=max(0.1,self.vo/self.p.R0/max(0.10,1.0-self.prev_u)+self.rng.uniform(-1.5,1.5))
            # Sample once per episode so the MDP observation is internally consistent.
            self._train_sched={
                'Vin1': self.p.Vin0+self.rng.uniform(-6,6),
                'Vin2': self.p.Vin0+self.rng.uniform(-8,8),
                'R1': float(self.rng.uniform(25,40)),
                'R2': float(self.rng.uniform(60,75)),
                'L1': self.p.L0*self.rng.uniform(0.90,1.10),
                'C1': self.p.C0*self.rng.uniform(0.90,1.10),
                'L2': self.p.L0*self.rng.uniform(0.90,1.10),
                'C2': self.p.C0*self.rng.uniform(0.90,1.10),
                'phase': self.rng.uniform(0,2*np.pi)
            }
        elif self.scenario in ('param','val_param'):
            self.vo=self.p.Vref-12.0
            self.iL=max(0.1,self.vo/self.p.R0/max(0.10,1.0-self.prev_u))
        self.prev_ev=self.p.Vref-self.vo
        return self._state()

    def _time(self): return self.t*self.p.Tc

    def disturbance(self):
        time=self._time(); frac=self.t/max(1,self.episode_steps)
        Vin,R,L,C=self.p.Vin0,self.p.R0,self.p.L0,self.p.C0
        if self.scenario=='load':
            if 0.30 <= frac < 0.65: R=25.0
            elif frac>=0.65: R=75.0
        elif self.scenario=='input':
            if 0.25 <= frac < 0.50: Vin=106.0
            elif 0.50 <= frac < 0.75: Vin=95.0
            Vin += 1.5*np.sin(2*np.pi*40.0*time)
        elif self.scenario=='param':
            # Fixed worst-case component mismatch used to test transient recovery under parameter uncertainty.
            L=self.p.L0*0.90; C=self.p.C0*1.12
        elif self.scenario=='val_load':
            if 0.30 <= frac < 0.65: R=30.0
            elif frac>=0.65: R=70.0
        elif self.scenario=='val_input':
            if 0.25 <= frac < 0.50: Vin=104.0
            elif 0.50 <= frac < 0.75: Vin=97.0
            Vin += 1.0*np.sin(2*np.pi*35.0*time)
        elif self.scenario=='val_param':
            if 0.30<=frac<0.60: L=self.p.L0*1.07; C=self.p.C0*0.95
            elif 0.60<=frac<0.80: L=self.p.L0*0.94; C=self.p.C0*1.08
            elif frac>=0.80: L=self.p.L0*1.03; C=self.p.C0*0.98
        elif self.scenario=='random':
            s=self._train_sched
            if frac < 0.30:
                Vin=self.p.Vin0; R=self.p.R0
            elif frac < 0.65:
                Vin=s['Vin1']; R=s['R1']; L=s['L1']; C=s['C1']
            else:
                Vin=s['Vin2']; R=s['R2']; L=s['L2']; C=s['C2']
            Vin += 1.0*np.sin(2*np.pi*30.0*time+s['phase'])
        return float(Vin),float(R),float(L),float(C)

    def _state(self):
        Vin,R,_,_=self.disturbance()
        ev=self.p.Vref-self.vo
        dev=ev-self.prev_ev
        return np.array([self.vo/self.p.Vref,self.iL/10.0,ev/self.p.Vref,dev/self.p.Vref,
                         Vin/self.p.Vin0,R/self.p.R0,self.prev_u],dtype=np.float32)

    def _integrate_rk4(self,u,Vin,R,L,C):
        n=max(1,int(round(self.p.Tc/self.p.h)))
        h=self.p.Tc/n
        i=float(self.iL); v=float(self.vo); omu=1.0-u
        for _ in range(n):
            k1i=(Vin-omu*v)/L; k1v=(omu*i-v/R)/C
            i2=i+0.5*h*k1i; v2=v+0.5*h*k1v
            k2i=(Vin-omu*v2)/L; k2v=(omu*i2-v2/R)/C
            i3=i+0.5*h*k2i; v3=v+0.5*h*k2v
            k3i=(Vin-omu*v3)/L; k3v=(omu*i3-v3/R)/C
            i4=i+h*k3i; v4=v+h*k3v
            k4i=(Vin-omu*v4)/L; k4v=(omu*i4-v4/R)/C
            i += (h/6.0)*(k1i+2.0*k2i+2.0*k3i+k4i)
            v += (h/6.0)*(k1v+2.0*k2v+2.0*k3v+k4v)
        self.iL=min(max(i,0.0),self.p.iL_clip)
        self.vo=min(max(v,self.p.vo_min),self.p.vo_max)

    def step(self,u):
        u=float(np.clip(u,self.p.u_min,self.p.u_max))
        Vin,R,L,C=self.disturbance()
        old_vo=self.vo
        du=u-self.prev_u
        self._integrate_rk4(u,Vin,R,L,C)
        ev=self.p.Vref-self.vo
        dev=ev-self.prev_ev
        # Three independent normalized objectives; 0.12 combines the two algebraically
        # redundant error/output-increment penalties of the original implementation.
        reward=-((ev/20.0)**2 + 0.12*(dev/20.0)**2 + 0.05*(du/0.10)**2)
        self.prev_ev=ev; self.prev_u=u; self.t+=1
        done=self.t>=self.episode_steps
        info={'vo':self.vo,'iL':self.iL,'u':u,'ev':ev,'Vin':Vin,'R':R,'bound_hit': self.vo>=self.p.vo_max-1e-9}
        return self._state(),float(reward),done,info

# -----------------------------
# Networks and replay
# -----------------------------
class ReplayBuffer:
    def __init__(self,sd,ad,max_size=250000):
        self.max_size=max_size; self.ptr=0; self.size=0
        self.s=np.zeros((max_size,sd),np.float32); self.a=np.zeros((max_size,ad),np.float32)
        self.r=np.zeros((max_size,1),np.float32); self.ns=np.zeros((max_size,sd),np.float32); self.d=np.zeros((max_size,1),np.float32)
    def add(self,s,a,r,ns,d):
        self.s[self.ptr]=s; self.a[self.ptr]=a; self.r[self.ptr]=r; self.ns[self.ptr]=ns; self.d[self.ptr]=d
        self.ptr=(self.ptr+1)%self.max_size; self.size=min(self.size+1,self.max_size)
    def sample(self,batch,device='cpu'):
        ind=np.random.randint(0,self.size,size=batch)
        return tuple(torch.as_tensor(arr[ind],device=device) for arr in (self.s,self.a,self.r,self.ns,self.d))

class Actor(nn.Module):
    def __init__(self,sd,ad):
        super().__init__(); self.net=nn.Sequential(nn.Linear(sd,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,ad),nn.Tanh())
    def forward(self,s): return self.net(s)

class TwinCritic(nn.Module):
    def __init__(self,sd,ad):
        super().__init__();
        self.q1=nn.Sequential(nn.Linear(sd+ad,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
        self.q2=nn.Sequential(nn.Linear(sd+ad,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a):
        x=torch.cat([s,a],1); return self.q1(x),self.q2(x)
    def q1_value(self,s,a): return self.q1(torch.cat([s,a],1))

class SingleCritic(nn.Module):
    def __init__(self,sd,ad):
        super().__init__(); self.q=nn.Sequential(nn.Linear(sd+ad,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a): return self.q(torch.cat([s,a],1))

class BaseAgent:
    def __init__(self,sd,ad,p,device='cpu'):
        self.p=p; self.device=device
        self.actor=Actor(sd,ad).to(device); self.actor_target=Actor(sd,ad).to(device); self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=1e-4)
    def scale(self,a): return (a+1.0)*0.5*(self.p.u_max-self.p.u_min)+self.p.u_min
    def select_action(self,s,noise_std=0.0):
        st=torch.as_tensor(s.reshape(1,-1),dtype=torch.float32,device=self.device)
        with torch.no_grad(): a=self.scale(self.actor(st)).cpu().numpy().ravel()[0]
        if noise_std: a += np.random.normal(0,noise_std)
        return float(np.clip(a,self.p.u_min,self.p.u_max))
    def soft(self,net,target,tau):
        for p,tp in zip(net.parameters(),target.parameters()): tp.data.mul_(1-tau).add_(tau*p.data)

class TD3Agent(BaseAgent):
    def __init__(self,sd,ad,p,device='cpu'):
        super().__init__(sd,ad,p,device); self.critic=TwinCritic(sd,ad).to(device); self.critic_target=TwinCritic(sd,ad).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict()); self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=1e-3); self.total_it=0
    def train(self,replay,batch=128,gamma=.99,tau=.005,policy_noise=.02,noise_clip=.05,policy_freq=2):
        self.total_it+=1; s,a,r,ns,d=replay.sample(batch,self.device)
        with torch.no_grad():
            noise=(torch.randn_like(a)*policy_noise).clamp(-noise_clip,noise_clip)
            na=(self.scale(self.actor_target(ns))+noise).clamp(self.p.u_min,self.p.u_max)
            tq1,tq2=self.critic_target(ns,na); y=r+(1-d)*gamma*torch.min(tq1,tq2)
        q1,q2=self.critic(s,a); cl=F.mse_loss(q1,y)+F.mse_loss(q2,y)
        self.critic_opt.zero_grad(); cl.backward(); self.critic_opt.step()
        al_val=0.0
        if self.total_it%policy_freq==0:
            aa=self.scale(self.actor(s)); al=-self.critic.q1_value(s,aa).mean()
            self.actor_opt.zero_grad(); al.backward(); self.actor_opt.step(); al_val=float(al.item())
            self.soft(self.critic,self.critic_target,tau); self.soft(self.actor,self.actor_target,tau)
        return float(cl.item()),al_val

class DDPGAgent(BaseAgent):
    def __init__(self,sd,ad,p,device='cpu'):
        super().__init__(sd,ad,p,device); self.critic=SingleCritic(sd,ad).to(device); self.critic_target=SingleCritic(sd,ad).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict()); self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=1e-3)
    def train(self,replay,batch=128,gamma=.99,tau=.005):
        s,a,r,ns,d=replay.sample(batch,self.device)
        with torch.no_grad():
            na=self.scale(self.actor_target(ns)).clamp(self.p.u_min,self.p.u_max)
            y=r+(1-d)*gamma*self.critic_target(ns,na)
        q=self.critic(s,a); cl=F.mse_loss(q,y)
        self.critic_opt.zero_grad(); cl.backward(); self.critic_opt.step()
        aa=self.scale(self.actor(s)); al=-self.critic(s,aa).mean()
        self.actor_opt.zero_grad(); al.backward(); self.actor_opt.step()
        self.soft(self.critic,self.critic_target,tau); self.soft(self.actor,self.actor_target,tau)
        return float(cl.item()),float(al.item())

# -----------------------------
# Conventional baselines
# -----------------------------
class PIController:
    def __init__(self,p,kp=.002,ki=6.0,antiwindup=False): self.p=p; self.kp=kp; self.ki=ki; self.aw=antiwindup; self.u0=.5; self.integral=0.0
    def reset(self): self.integral=0.0
    def action(self,vo,iL,prev_u):
        e=self.p.Vref-vo; cand_int=self.integral+e*self.p.Tc; raw=self.u0+self.kp*e+self.ki*cand_int
        if self.aw:
            sat=np.clip(raw,self.p.u_min,self.p.u_max)
            if (raw==sat) or (raw>self.p.u_max and e<0) or (raw<self.p.u_min and e>0): self.integral=cand_int
        else: self.integral=cand_int
        raw=self.u0+self.kp*e+self.ki*self.integral
        return float(np.clip(raw,self.p.u_min,self.p.u_max))

class SMCController:
    def __init__(self,p,k=.06,lam=.8): self.p=p; self.k=k; self.lam=lam; self.prev_e=0; self.u0=.5
    def reset(self): self.prev_e=0
    def action(self,vo,iL,prev_u):
        e=(self.p.Vref-vo)/self.p.Vref; de=e-self.prev_e; s=de+self.lam*e; self.prev_e=e
        return float(np.clip(self.u0+.25*e+self.k*np.tanh(8*s),self.p.u_min,self.p.u_max))

# -----------------------------
# Training / evaluation
# -----------------------------
def train_agent(kind,seed,out_dir,episodes=80,steps=1000):
    set_seed(seed); p=BoostParams(); sd,ad=7,1
    agent=TD3Agent(sd,ad,p) if kind=='TD3' else DDPGAgent(sd,ad,p)
    replay=ReplayBuffer(sd,ad,max_size=250000); env=BoostConverterEnv(p,'random',steps,seed=seed+1234)
    warmup=1000; batch=128; global_step=0; rows=[]
    for ep in range(1,episodes+1):
        s=env.reset('random'); er=0.; vos=[]; dus=[]; prev=env.prev_u; cls=[]; als=[]
        for _ in range(steps):
            u=np.random.uniform(p.u_min,p.u_max) if global_step<warmup else agent.select_action(s,noise_std=.03)
            ns,r,done,info=env.step(u); replay.add(s,[u],r,ns,float(done)); s=ns; er+=r; vos.append(info['vo']); dus.append(abs(info['u']-prev)); prev=info['u']; global_step+=1
            if replay.size>batch:
                cl,al=agent.train(replay,batch); cls.append(cl); als.append(al)
            if done: break
        rmse=float(np.sqrt(np.mean((np.asarray(vos)-p.Vref)**2))); rows.append([ep,er/steps,rmse,float(np.mean(dus)),float(np.mean(cls[-50:])) if cls else 0,float(np.mean(als[-50:])) if als else 0])
        if ep%max(1,episodes//8)==0: print(f'{kind} seed={seed} ep={ep}/{episodes} reward={rows[-1][1]:.3f} rmse={rmse:.3f}',flush=True)
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'training_log.csv',['episode','average_reward','voltage_rmse','average_duty_variation','critic_loss','actor_loss'],rows)
    torch.save(agent.actor.state_dict(),out/f'{kind.lower()}_actor.pt')
    return agent,rows

def run_controller(name,agent,scenario,steps=4000,h=None,pi_gains=(.002,6.0),smc_gains=(.06,.8)):
    p=BoostParams(h=BoostParams.h if h is None else h); env=BoostConverterEnv(p,scenario,steps,seed=999); s=env.reset(scenario)
    if name=='PI': ctrl=PIController(p,*pi_gains,False); ctrl.reset()
    elif name=='PI-AW': ctrl=PIController(p,*pi_gains,True); ctrl.reset()
    elif name=='SMC': ctrl=SMCController(p,*smc_gains); ctrl.reset()
    else: ctrl=None
    out={'t':[],'vo':[],'u':[],'iL':[],'bound':[]}
    for _ in range(steps):
        if name in ('TD3','DDPG'): u=agent.select_action(s,0.0)
        else: u=ctrl.action(env.vo,env.iL,env.prev_u)
        s,r,done,info=env.step(u)
        out['t'].append(env._time()); out['vo'].append(info['vo']); out['u'].append(info['u']); out['iL'].append(info['iL']); out['bound'].append(info['bound_hit'])
        if done: break
    return {k:np.asarray(v) for k,v in out.items()}

SCENARIO_META={
    'load': {'first':0.60,'final':1.30},
    'input': {'first':0.50,'final':1.50},
    'param': {'first':0.0,'final':0.0},
}

def metrics(res,scenario,vref=200.0):
    t=res['t']; vo=res['vo']; first=SCENARIO_META[scenario]['first']; final=SCENARIO_META[scenario]['final']
    md=t>=first; mf=t>=final
    if not np.any(md): md=np.ones_like(t,dtype=bool)
    rmse=float(np.sqrt(np.mean((vo[md]-vref)**2)))
    peak=float(np.max(vo[md])); overs=max(0.,(peak-vref)/vref*100)
    bound=bool(np.any(res['bound'][md]))
    final_window=t>=max(t[-1]-.2,0)
    fw=float(np.mean(np.abs(vo[final_window]-vref)))
    band=.05*vref; rec=np.nan
    ids=np.where(mf)[0]
    for idx in ids:
        if np.all(np.abs(vo[idx:]-vref)<=band): rec=float((t[idx]-final)*1000); break
    return {'rmse':rmse,'overshoot':overs,'recovery_ms':rec,'fw_error':fw,'bound_hit':bound,'peak':peak,'max_iL':float(np.max(res['iL']))}

def tune_baselines():
    """Select conventional-controller gains only on validation schedules, not on final test cases."""
    def validation_score(name, pi_gains=(5e-5,0.03), smc_gains=(0.11,0.5)):
        vals=[]
        for sc in ('val_load','val_input','val_param'):
            r=run_controller(name,None,sc,steps=1600,h=5e-5,pi_gains=pi_gains,smc_gains=smc_gains)
            vo=r['vo']; md=np.arange(len(vo))>=int(0.25*len(vo))
            rmse=float(np.sqrt(np.mean((vo[md]-200.0)**2)))
            overs=max(0.0,(float(np.max(vo[md]))-200.0)/2.0)
            bound=bool(np.any(r['bound'][md]))
            vals.append(rmse+0.3*overs+25.0*float(bound))
        return float(np.mean(vals))
    pi_candidates=[(kp,ki) for kp in (5e-5,1e-4,2e-4,3e-4,5e-4,1e-3) for ki in (0.005,0.01,0.03,0.05,0.1)]
    common=min(pi_candidates,key=lambda z:0.5*(validation_score('PI',pi_gains=z)+validation_score('PI-AW',pi_gains=z)))
    smc_candidates=[(k,lam) for k in (0.03,0.05,0.07,0.09,0.11) for lam in (0.3,0.5,0.8,1.1)]
    best_smc=min(smc_candidates,key=lambda z:validation_score('SMC',smc_gains=z))
    return common,best_smc

def exact_paired_permutation(a,b):
    # Two-sided exact sign-flip test of paired differences; n=6 -> 64 permutations.
    d=np.asarray(a,float)-np.asarray(b,float); obs=abs(np.mean(d)); vals=[]
    for signs in itertools.product((-1,1),repeat=len(d)): vals.append(abs(np.mean(d*np.asarray(signs))))
    return float(np.mean(np.asarray(vals)>=obs-1e-12))

def plot_results(out_dir,train_logs,agents_by_seed,representative_seed,pi_gains,smc_gains):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    # Fig 8: mean/std training RMSE for both algorithms
    fig,ax=plt.subplots(figsize=(8.4,5.2))
    for kind in ('TD3','DDPG'):
        arr=np.array([[r[2] for r in train_logs[(kind,s)]] for s in sorted(agents_by_seed[kind])],float)
        mean=arr.mean(0); std=arr.std(0,ddof=1); x=np.arange(1,len(mean)+1); mean=moving_average(mean,7); std=moving_average(std,7)
        ax.plot(x,mean,label=f'{kind} mean RMSE'); ax.fill_between(x,np.maximum(0,mean-std),mean+std,alpha=.16)
    ax.set_xlabel('Training episode'); ax.set_ylabel('Episode voltage RMSE (V)'); ax.grid(True,linestyle=':'); ax.legend(); fig.tight_layout()
    fig.savefig(out/'fig8_training_convergence_six_seed.jpeg',dpi=300,bbox_inches='tight'); plt.close(fig)

    td3=agents_by_seed['TD3'][representative_seed]; ddpg=agents_by_seed['DDPG'][representative_seed]
    methods=[('PI',None),('PI-AW',None),('SMC',None),('DDPG',ddpg),('TD3',td3)]
    plot_methods=[('PI',None),('SMC',None),('DDPG',ddpg),('TD3',td3)]
    allres={}
    for sc in ('load','input','param'):
        allres[sc]={m:run_controller(m,a,sc,pi_gains=pi_gains,smc_gains=smc_gains) for m,a in methods}

    # Fig 9 output responses
    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
    titles={'load':'(a) Load disturbance','input':'(b) Input-voltage fluctuation','param':'(c) Parameter-mismatch recovery'}
    for ax,sc in zip(axs,('load','input','param')):
        for m,_ in plot_methods: ax.plot(allres[sc][m]['t'],allres[sc][m]['vo'],label=m,linewidth=1.2)
        ax.axhline(200,linestyle='--',linewidth=.9); ax.set_ylabel('Output voltage (V)'); ax.set_title(titles[sc]); ax.grid(True,linestyle=':')
    axs[-1].set_xlabel('Time (s)'); axs[0].legend(ncol=5,fontsize=8); fig.tight_layout(); fig.savefig(out/'fig9_dynamic_response_comparison.jpeg',dpi=300,bbox_inches='tight'); plt.close(fig)

    # Fig 10 zoomed output response around reference; y range based on non-PI robust methods, but include all.
    fig,axs=plt.subplots(3,1,figsize=(9,8.5),sharex=True)
    for ax,sc in zip(axs,('load','input','param')):
        for m,_ in plot_methods: ax.plot(allres[sc][m]['t'],allres[sc][m]['vo'],label=m,linewidth=1.1)
        ax.axhline(200,linestyle='--',linewidth=.8); ax.set_ylim(170,235); ax.set_ylabel('Voltage (V)'); ax.set_title(titles[sc]); ax.grid(True,linestyle=':')
    axs[-1].set_xlabel('Time (s)'); axs[0].legend(ncol=5,fontsize=8); fig.tight_layout(); fig.savefig(out/'fig10_output_voltage_response_details.jpeg',dpi=300,bbox_inches='tight'); plt.close(fig)

    # Fig 11 duty trajectories
    fig,axs=plt.subplots(3,1,figsize=(9,8.5),sharex=True)
    for ax,sc in zip(axs,('load','input','param')):
        for m,_ in plot_methods: ax.plot(allres[sc][m]['t'],allres[sc][m]['u'],label=m,linewidth=1.0)
        ax.set_ylabel('Duty cycle'); ax.set_title(titles[sc]); ax.grid(True,linestyle=':')
    axs[-1].set_xlabel('Time (s)'); axs[0].legend(ncol=5,fontsize=8); fig.tight_layout(); fig.savefig(out/'fig11_duty_cycle_trajectories.jpeg',dpi=300,bbox_inches='tight'); plt.close(fig)

    # Fig 12 abs error
    fig,axs=plt.subplots(3,1,figsize=(9,8.5),sharex=True)
    for ax,sc in zip(axs,('load','input','param')):
        for m,_ in plot_methods: ax.plot(allres[sc][m]['t'],np.abs(allres[sc][m]['vo']-200),label=m,linewidth=1.0)
        ax.set_ylabel('|Error| (V)'); ax.set_title(titles[sc]); ax.grid(True,linestyle=':')
    axs[-1].set_xlabel('Time (s)'); axs[0].legend(ncol=5,fontsize=8); fig.tight_layout(); fig.savefig(out/'fig12_absolute_tracking_error_trajectories_zoomed.jpeg',dpi=300,bbox_inches='tight'); plt.close(fig)
    return allres

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='revised_experiment'); ap.add_argument('--episodes',type=int,default=20); ap.add_argument('--steps',type=int,default=500); ap.add_argument('--seeds',nargs='+',type=int,default=list(range(6))); args=ap.parse_args()
    torch.set_num_threads(1); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    # Select baseline gains on validation schedules that differ from final tests.
    pi_gains,smc_gains=tune_baselines()
    print('Validation-selected PI/PI-AW gains',pi_gains,'SMC',smc_gains,flush=True)
    agents={'TD3':{},'DDPG':{}}; logs={}
    t0=time.time()
    for kind in ('TD3','DDPG'):
        for seed in args.seeds:
            agent,rows=train_agent(kind,seed,out/f'training/{kind.lower()}_seed{seed}',args.episodes,args.steps); agents[kind][seed]=agent; logs[(kind,seed)]=rows
    print('Training seconds',time.time()-t0,flush=True)

    # Evaluate every trained seed on all three scenarios.
    raw=[]
    for kind in ('TD3','DDPG'):
        for seed,agent in agents[kind].items():
            for sc in ('load','input','param'):
                res=run_controller(kind,agent,sc,pi_gains=pi_gains,smc_gains=smc_gains); m=metrics(res,sc)
                raw.append([kind,seed,sc,m['rmse'],m['overshoot'],m['recovery_ms'],m['fw_error'],m['bound_hit'],m['max_iL']])
    write_csv(out/'six_seed_raw.csv',['algorithm','seed','scenario','rmse','overshoot','recovery_ms','final_window_error','bound_hit','max_iL'],raw)

    # aggregate and p-values TD3 vs DDPG
    summary=[]
    for sc in ('load','input','param'):
        vals={}
        for kind in ('TD3','DDPG'):
            rows=[r for r in raw if r[0]==kind and r[2]==sc]; vals[kind]=rows
            arr=np.array([[r[3],r[4],r[6]] for r in rows],float)
            summary.append([kind,sc,*arr.mean(0),*arr.std(0,ddof=1)])
        p_rmse=exact_paired_permutation([r[3] for r in vals['TD3']],[r[3] for r in vals['DDPG']])
        p_ov=exact_paired_permutation([r[4] for r in vals['TD3']],[r[4] for r in vals['DDPG']])
        p_fw=exact_paired_permutation([r[6] for r in vals['TD3']],[r[6] for r in vals['DDPG']])
        write_csv(out/f'pvalues_{sc}.csv',['metric','p_exact'],[['RMSE',p_rmse],['Overshoot',p_ov],['Final-window error',p_fw]])
    write_csv(out/'six_seed_summary.csv',['algorithm','scenario','rmse_mean','overshoot_mean','fw_mean','rmse_std','overshoot_std','fw_std'],summary)

    # Select a representative TD3 seed as the medoid relative to the component-wise median RMSE vector.
    td3_medians={sc:np.median([r[3] for r in raw if r[0]=='TD3' and r[2]==sc]) for sc in ('load','input','param')}
    rep=min(args.seeds,key=lambda seed:sum((next(r[3] for r in raw if r[0]=='TD3' and r[1]==seed and r[2]==sc)-td3_medians[sc])**2 for sc in td3_medians))
    print('Representative seed',rep,flush=True)
    allres=plot_results(out/'figures',logs,agents,rep,pi_gains,smc_gains)

    # main table: representative-seed RL + fixed conventional baselines
    td3=agents['TD3'][rep]; ddpg=agents['DDPG'][rep]
    method_agents=[('PI',None),('PI-AW',None),('SMC',None),('DDPG',ddpg),('TD3',td3)]
    for sc in ('load','input','param'):
        rows=[]
        for m,a in method_agents:
            rr=allres[sc][m]; met=metrics(rr,sc)
            rows.append([m,met['rmse'],met['overshoot'],met['recovery_ms'],met['fw_error'],met['bound_hit'],met['max_iL']])
        write_csv(out/f'table_{sc}.csv',['Method','Post-disturbance RMSE (V)','Overshoot (%)','Final-transition recovery (ms)','Final-window error (V)','Bound hit','Max iL (A)'],rows)

    # numerical integration sensitivity for representative TD3 actor
    sens=[]; hs=[5e-5,2.5e-5,1e-5]
    for h in hs:
        for sc in ('load','input','param'):
            rr=run_controller('TD3',td3,sc,h=h,pi_gains=pi_gains,smc_gains=smc_gains); met=metrics(rr,sc); sens.append([h*1e6,sc,met['rmse'],met['overshoot'],met['fw_error']])
    write_csv(out/'integration_step_sensitivity.csv',['RK4 internal step (us)','Scenario','RMSE (V)','Overshoot (%)','Final-window error (V)'],sens)
    fig,ax=plt.subplots(figsize=(7.4,4.8))
    for sc in ('load','input','param'):
        ss=[r for r in sens if r[1]==sc]; ax.plot([r[0] for r in ss],[r[2] for r in ss],marker='o',label=sc)
    ax.set_xlabel('RK4 internal integration step (μs)'); ax.set_ylabel('Post-disturbance RMSE (V)'); ax.set_xscale('log'); ax.invert_xaxis(); ax.grid(True,linestyle=':'); ax.legend(); fig.tight_layout(); fig.savefig(out/'figures/fig13_integration_step_sensitivity.jpeg',dpi=300,bbox_inches='tight'); plt.close(fig)

    # save configuration
    with open(out/'config.txt','w') as f:
        f.write(f'controller_period_us=500\ninternal_RK4_step_us=50\nepisodes={args.episodes}\nsteps_per_episode={args.steps}\nseeds={args.seeds}\n')
        f.write(f'pi_common_kp={pi_gains[0]}\npi_common_ki={pi_gains[1]}\nsmc_k={smc_gains[0]}\nsmc_lambda={smc_gains[1]}\n')
        f.write('actor_lr=1e-4\ncritic_lr=1e-3\ngamma=0.99\ntau=0.005\nexploration_noise=0.03\ntd3_target_noise=0.02\ntd3_noise_clip=0.05\ntd3_policy_delay=2\n')
    print('Done:',out,flush=True)

if __name__=='__main__': main()
