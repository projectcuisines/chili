### Quick notes for intercomparison data

#### Model Scope & Cases
- **Included Species**: Our model considers **H2O only**.
- **Hydrogen (H)**: We calculated the **nominal** case and the **low / mid / high** cases.
- **Carbon (C)**: All calculations were performed with **zero carbon (C = 0)**.
  - Data files are named as `Hlow/mid/high_Clow`.
  - Files for `Cmid` and `Chigh` are **not included**.

#### Missing Cases / Exclusions
- **TRAPPIST-1 alpha**:
  Data for this case is **not submitted**. Under the provided initial conditions, the resulting surface temperature exceeds the applicable temperature range of our model.
  
#### Additional Cases
- The `aux/` folder contains additional simulation cases. Currently, this is one case in which the assumed melt-trapping fraction is increased to 40%, from the 1% baseline assumed in the main simulations (informed by Elkins-Tanton 2008).

