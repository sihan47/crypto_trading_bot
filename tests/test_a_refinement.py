import unittest
import numpy as np
import pandas as pd
from research.a_refinement import frames,daily_atr
from research.impulse_scan import discovery
from research.archetype_tournament import simulate


class RefinementTests(unittest.TestCase):
    def fixture(self,n=800):
        ix=pd.date_range('2024-01-01',periods=n,freq='15min',tz='UTC')
        raw=pd.DataFrame({'open':100.,'high':100.1,'low':99.9,'close':100.,'volume':1e6},index=ix)
        f=pd.DataFrame({'signal':False,'stop':90.,'sigma':.1,'daily_atr':10.,'risk_scale':1.,
                        'limit':np.nan,'target':np.nan,'regime':'bull','vol_regime':'low','exit_signal':False},index=ix[::4])
        f.loc[f.index[0],'signal']=True
        return raw,f

    def test_no_time_stop_and_regime_exit_next_bar(self):
        raw,f=self.fixture()
        _,old,_=simulate(raw,f,'A')
        self.assertEqual(old.reason.iloc[0],'time_exit')
        _,new,_=simulate(raw,f,'A',no_time_stop=True,chandelier=True)
        self.assertGreater(new.hours.iloc[0],168)
        f.loc[f.index[3],'exit_signal']=True
        _,tr,_=simulate(raw,f,'A',no_time_stop=True,chandelier=True)
        self.assertEqual(tr.exit_time.iloc[0],str(raw.index[13]))
        self.assertEqual(tr.reason.iloc[0],'regime_exit')

    def test_risk_scaling_and_stop_uses_prior_bar(self):
        raw,f=self.fixture(16)
        _,one,_=simulate(raw,f,'A')
        f['risk_scale']=.5
        _,half,_=simulate(raw,f,'A')
        self.assertAlmostEqual(one.entry_notional.iloc[0]/2,half.entry_notional.iloc[0])
        raw.loc[raw.index[1],'high']=150
        _,tr,_=simulate(raw,f,'A',no_time_stop=True,chandelier=True)
        self.assertEqual(tr.exit_time.iloc[0],str(raw.index[2]))

    def test_closed_daily_atr_and_prefix(self):
        d=discovery().iloc[:6000]
        full=frames(d);short=frames(d.iloc[:5500])
        for name in full:pd.testing.assert_frame_equal(full[name].iloc[:5500],short[name])
        atr=daily_atr(d)
        # Within a UTC day only the last completed daily ATR is available.
        valid=atr.dropna();day=valid.loc[valid.index.floor('D')==valid.index[-1].floor('D')]
        self.assertEqual(day.nunique(),1)


if __name__=='__main__':unittest.main()
