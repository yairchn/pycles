cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
cimport Thermodynamics
cimport TimeStepping

cdef class ForcingGCMVarying:
    cdef:
        bint gcm_profiles_initialized
        int t_indx
        double [:] ug
        double [:] vg
        double [:] subsidence
        double [:] temp_dt_hadv
        double [:] temp_dt_fino
        double [:] temp_dt_resid
        double [:] shum_dt_vadv
        double [:] shum_dt_hadv
        double [:] shum_dt_resid


        double [:] u_dt_hadv
        double [:] u_dt_vadv
        double [:] u_dt_cof
        double [:] u_dt_pres
        double [:] u_dt_tot

        double [:] v_dt_hadv
        double [:] v_dt_vadv
        double [:] v_dt_cof
        double [:] v_dt_pres
        double [:] v_dt_tot

        double [:] p_gcm
        double [:] rho_gcm
        double [:] rho_half_gcm
        double coriolis_param
        str file
        double lat
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,  TimeStepping.TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa)
    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa)
