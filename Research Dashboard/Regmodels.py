import numpy as np
import pandas as pd
from scipy.stats import t, f

class OLS:
    def __init__(self, y, x):
        df = pd.merge(y, x, left_index=True, right_index=True)
        df = df.dropna()
        self.y = df.iloc[:,0]
        x = df.iloc[:,1:]

        cats = x.select_dtypes(include=['object', 'category']).columns
        if len(cats)>0:
            dummies = pd.get_dummies(x[cats], drop_first=True,dtype=int)
            numerical = x.select_dtypes(exclude=['object','category']).columns
            self.x = pd.concat([x[numerical], dummies], axis=1)
        else:
            self.x = df.iloc[:,1:]
        
        print(self.x)

    def fit(self, constant = True, covtype = 'standard'):
        if covtype not in ('standard', 'white', 'hc1', 'hc2'):
            raise ValueError(f"Invalid covtype: {covtype}.")
        
        ones = np.ones((self.x.shape[0],1))
        features = np.array(self.x.columns.tolist())
        x = np.array(self.x)
        df_model = x.shape[1]

        if constant:
            features = np.array(['Constant'] + features.tolist())
            x = np.hstack((ones, x))

        df_residual = len(self.y) - x.shape[1]
        beta = np.linalg.inv(x.T @ x) @ x.T @ self.y 

        yhat = x @ beta
        e = self.y - yhat
        ssr = e.T @ e
        s2 = ssr / df_residual

        if covtype == 'standard':
            varcov = s2 * np.linalg.inv(x.T @ x)
        elif covtype == 'white':
            varcov = self.white(x, e)
        elif covtype == 'hc1':
            varcov = self.hc1(x, e)
        elif covtype == 'hc2':
            varcov = self.hc2(x, e)

        diag = np.diagonal(varcov)
        se = np.sqrt(diag)
        t_stat = (beta / se)
        p = 2 * t.sf(np.abs(t_stat), df = df_residual)

        sst = self.y.T @ (np.eye(x.shape[0]) - ones @ np.linalg.pinv(ones.T @ ones) @ ones.T) @ self.y

        R2 = 1 - (ssr/sst) 
        adjR2 = 1- ((ssr/df_residual)/(sst/(len(self.y)- 1)))
        uncR2 = 1-(ssr/(self.y @ self.y))

        output = pd.DataFrame({
            'Variable': features,
            'coef': [f'{x:.4f}' for x in beta],
            'std. err': [f'{x:.4f}' for x in se],
            't': [f'{x:.4f}' for x in t_stat],
            'p-value': [f'{x:.4f}' for x in p]
            }
        )
        if constant:
            e_r = self.y - ones @ np.linalg.inv(ones.T @ ones) @ ones.T @ self.y 
            ssr_restr = e_r.T @ e_r
            F = ((ssr_restr - ssr) / df_model) / (ssr / df_residual)
        else:
            F = ((self.y.T @ self.y - ssr) / df_model) / (ssr / df_residual) 
        
        p_f = 1 - f.cdf(F, df_model, df_residual)

        self.features = features
        self.output = output
        self.residuals = e
        self.fitted = yhat
        self.R2 = R2
        self.adjustedR2 = adjR2
        self.uncenteredR2 = uncR2
        self.SSR = ssr
        self.SST = sst
        self.varcov = varcov
        self.dfResid = df_residual
        self.dfModel = df_model
        self.F = F
        self.probF = p_f
        self.n = len(self.y)
        self.covtype = covtype

        return output
    
    def white(self, X, e):
        Sigma_hat = np.diag(e**2)
        return np.linalg.pinv(X.T @ X) @ X.T @ Sigma_hat @ X @ np.linalg.pinv(X.T @ X)
    
    def hc1(self, X, e):
        n, k = X.shape
        return n/(n-k) * self.white(X,e)
    
    def hc2(self, X, e):
        H_diag = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
        Sigma_hat = np.diag((e / (1 - H_diag))**2)
        return np.linalg.pinv(X.T @ X) @ X.T @ Sigma_hat @ X @ np.linalg.pinv(X.T @ X)

    def output_to_latex(self):
        ltex = r"\begin{table}[ht!]" + "\n"
        ltex += r"\centering" + "\n"
        ltex += r"\caption{OLS Regression Results}" + "\n"
        ltex += r"\label{tab:regression_results}" + "\n"
        ltex += r"\begin{tabular}{lllll}" + "\n"  
        ltex += r"\toprule" + "\n"
        ltex += f"Model: & OLS & & R-squared: & {self.R2:.4f} \\\\" + "\n"
        ltex += f"No. Observations: & {self.n} & & Adj. R-squared: & {self.adjustedR2:.4f} \\\\" + "\n"
        ltex += f"Df Residuals: & {self.dfResid} & & Unc. R-squared: & {self.uncenteredR2:.4f} \\\\" + "\n"
        ltex += f"Df Model: & {self.dfModel} & & F-statistic: & {self.F:.3f} \\\\" + "\n"
        ltex += f"Covariance Type: & {self.covtype} & & Prob(F-statistic): & {self.probF:.3f} \\\\" + "\n"
        ltex += r"\midrule" + "\n"
        ltex += r"Variable & Coefficient & Std. Error & t-statistic & p-value \\" + "\n"
        ltex += r"\hline" + "\n"
        for label, (_, row) in zip(self.features, self.output.iterrows()):
            ltex += (
                f"{label} & {row['coef']:.4f} & {row['std. err']:.3f} & {row['t']:.3f} & "
                f"{row['p-value']:.3f} \\\\" + "\n"
            )
        ltex += r"\bottomrule" + "\n"
        ltex += r"\end{tabular}" + "\n"
        ltex += r"\end{table}" + "\n"
        return ltex