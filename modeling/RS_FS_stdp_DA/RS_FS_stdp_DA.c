#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define dim 1000
#define dimmexc 800
#define diminh 200

#define Pi 3.1415

double tetaaa(double);
double doubgauss(double t, double t0, double T1, double T2, double amp);
double heaviside(double x);
double x_nmda(double);
double rand_lognormal(double, double);

FILE *timecourse, *rasterexc, *rasterinh, *conn_0, *conn_f;

int t, i, r, j, s;

int gij_connections[dim][dim];
double gij_weights[dim][dim];
double eligibility[dim][dim];
double lastupdate[dim][dim];

int gijintEE[dimmexc][dimmexc];
int gijintEI[dimmexc][diminh];
int gijintIE[diminh][dimmexc];
int gijintII[diminh][diminh];

int main()
{

    rasterexc = fopen("rasterplot_EXc_withstimuli_4.dat", "w");
    rasterinh = fopen("rasterplot_Inh_withstimuli_4.dat", "w");
    timecourse = fopen("traces_withstimuli_4.dat", "w");

    conn_0 = fopen("weight_icB.dat", "r");
    conn_f = fopen("weight_fin4.dat", "w");

    // series of rewards, a=first, b=second etc...
    int number_of_rewards = 4;
    double time_between_rewards = 4; // 4 seconds
    double rewards[number_of_rewards];
    rewards[0] = 1;  // +reward
    rewards[1] = 1;  //+reward
    rewards[2] = -1; //-reward
    rewards[3] = -1; //-reward

    double BIN = .001; // sampling rate (seconds)

    double inputrateonexc0; // rate of external drive constant in time
    double inputextern = 0;
    double inputextern_inh = 0;
    // features of external input
    double T1, T2, amp, t0;
    t0 = 10;   // time of appearance
    T1 = 0.05; // rise time (seconds)
    T2 = 0.2;  // decay time (seconds)
    amp = 4;   // amplitude (Hz)

    inputrateonexc0 = 1.5;

    /*dimulation duration*/

    double tstop = 28.;
    double transientt = 0.;
    double stepp = 0.005 * 1.e-3; /*until 0.002*/
                                  /*VARIABLES DEFINITION*/

    double v[dim], vold[dim];
    double tolsp[dim];
    double told;
    int hui, huj;
    double GE_gabaa[dimmexc], GE_ampa[dimmexc], GI_gabaa[diminh], GI_ampa[diminh], ww[dimmexc], rexc, rinh;

    double GE_nmda[dimmexc], GI_nmda[diminh];
    double GE_gabab[dimmexc], GI_gabab[diminh], QE_gabab[dimmexc], QI_gabab[diminh];

    double p_gabaB[diminh], q_gabaB[diminh];

    double spike_flag[dim]; // keeps track of last spike time for all neurons
    double rate, ratebinn;
    double timet;
    double randnumb;
    double delay = 0.5 * 1.e-3;
    double kmedio;

    double inputrateonexc;
    double inputrateoninh = 0.;

    // STDP variables and parameters
    double deltaij = 0;
    double dd = 0.05;
    double pp = 0.1;
    double taup = 10 * 1.e-3;
    double taum = taup;
    double wm = 1.1; // wmax
    double em = 10;
    double wmin = 0.0;

    double tauc = 1;
    double dopamine;

    double fracin = 0.2;
    int ni = floor(fracin * dim);
    int ne = floor(dim * (1 - fracin));

    /*connection parameters*/

    double pconn = 0.05;
    pconn = pconn * 10000 / dim;

    double muw = 0;
    double sigmaw = 0.;

    double Ee = 0 * 1.e-3;
    double Ei = -80 * 1.e-3;

    /*neurons parameters*/

    double glRS = 15. * 1.e-9;
    double CmRS = 200 * 1.e-12;
    double bRS = 0 * 1.e-12;
    double aRS = 0.e-9;
    double tauw = 100 * 1.e-3;
    double deltaRS = 2 * 1.e-3;
    double ElRS = -65 * 1.e-3;

    double ElFS = -65 * 1.e-3;
    double glFS = 15 * 1.e-9;
    double CmFS = 200 * 1.e-12;
    double bFS = 0 * 1.e-12;
    double aFS = 0 * 1.e-9;
    double twFS = 500;
    double deltaFS = 0.5 * 1.e-3;

    // synaptic dynamics

    // AMPA
    double tau_decay_ampa = 5 * 1.e-3;
    double g_ampa = 1.5 * 1.e-9;

    // GABA_A
    double tau_decay_gabaa = 5 * 1.e-3;
    double g_gabaa = 5 * 1.e-9;

    // NMDA
    double Enmda = 0;
    double tau_decay_nmda = 75 * 1.e-3;
    double g_nmda = 0.012 * 1.e-9;

    // GABAB current
    double EgabaB = -90 * 1.e-3;
    double alpha_gabaB = 1;
    double tau_decay_gabaB = 160 * 1.e-3;
    double tau_rise_gabaB = 90 * 1.e-3;
    double deltaq_gabaB = 0.1;
    double g_gabaB = 1. * 1.e-9;

    double Trefr = 5 * 1.e-3;

    double vres = -65 * 1.e-3;
    double thr = -50 * 1.e-3;
    double vspike = -30 * 1.e-3;

    for (i = 0; i < dim; i++)
    {
        for (r = 0; r < dim; r++)
        {
            randnumb = rand() / (RAND_MAX + 1.0);

            gij_connections[i][r] = 0;
            if (randnumb < pconn)
            {
                gij_connections[i][r] = 1;
            }
        }
    }

    for (i = 0; i < dim; i++)
    {
        for (r = 0; r < dim; r++)
        {

            gij_weights[i][r] = rand_lognormal(muw, sigmaw);
        }
    }

    for (i = 0; i < dimmexc; i++)
    {
        for (j = 0; j < dimmexc; j++)
        {
            if (gij_weights[i][j] < wmin)
            {
                gij_weights[i][j] = wmin;
            }
            if (gij_weights[i][j] > wm)
            {
                gij_weights[i][j] = wm;
            }
        }
    }

    for (j = 0; j < ne; j++)
    {

        v[j] = (-80 + 10 * (0.5 - rand() / (RAND_MAX + 1.0))) * 1.e-3;
        ;
        ww[j] = 0.;
        tolsp[j] = 0;
        spike_flag[j] = 10000000;
    }

    for (j = ne; j < dim; j++)
    {

        v[j] = (-80 + 10 * (0.5 - rand() / (RAND_MAX + 1.0))) * 1.e-3;
        tolsp[j] = 0;
        spike_flag[j] = 10000000;
    }

    /*TIME SIMULATION*/

    timet = 0;
    told = 0;

    while (timet < tstop)
    {

        for (i = 0; i < dim; i++)
        {
            vold[i] = v[i];
        }

        timet += stepp;
        inputextern = 0;

        dopamine = rewards[0] * doubgauss(timet, t0, .05, .05, 1) + rewards[1] * doubgauss(timet, t0 + time_between_rewards, .05, .05, 1) + rewards[2] * doubgauss(timet, t0 + time_between_rewards * 2, .05, .05, 1) + rewards[3] * doubgauss(timet, t0 + time_between_rewards * 3, .05, .05, 1);

        inputrateonexc = inputrateonexc0;

        if (timet > told + BIN)
        {
            told = timet;
            // printf("%lf\n",timet);
            fprintf(timecourse, "%lf %lf %lf    %E  %E %E %E %E  %E  %E  %E\n", timet, (rexc / dimmexc) / BIN, (rinh / diminh) / BIN, inputrateonexc, (v[0] - EgabaB) * g_gabaB * GE_gabab[0], (v[0] - Ei) * GE_gabaa[0], -(v[0] - Ee) * 4 * g_nmda * x_nmda(v[0]) * GE_nmda[0], -(v[0] - Ee) * 4 * GE_ampa[0], gij_weights[0][150], eligibility[0][150], dopamine);

            rexc = 0;
            rinh = 0;
        }

        for (i = 0; i < dimmexc; i++)
        {

            randnumb = rand() / (RAND_MAX + 1.0);

            if (randnumb < stepp * inputrateonexc * pconn * dimmexc)
            {
                GE_ampa[i] += g_ampa;
            }

            randnumb = rand() / (RAND_MAX + 1.0);

            if (randnumb < stepp * inputrateoninh * pconn * diminh)
            {
                GE_gabaa[i] += g_gabaa;
            }

            randnumb = rand() / (RAND_MAX + 1.0);

            if (randnumb < stepp * inputextern * pconn * dimmexc)
            {
                GE_ampa[i] += g_ampa;
            }

            if (randnumb < stepp * inputextern_inh * pconn * dimmexc)
            {
                GE_gabaa[i] += g_gabaa;
            }

            v[i] += (glRS * (ElRS - v[i]) / CmRS + glRS * deltaRS * exp(((v[i] - thr) / (deltaRS))) / CmRS + GE_ampa[i] * (Ee - v[i]) / CmRS + GE_gabaa[i] * (Ei - v[i]) / CmRS + x_nmda(v[i]) * g_nmda * GE_nmda[i] * (Ee - v[i]) / CmRS - ww[i] / CmRS + g_gabaB * GE_gabab[i] * (EgabaB - v[i]) / CmRS) * stepp * tetaaa(timet - tolsp[i] - Trefr);

            ww[i] += (-ww[i] + aRS * (v[i] - ElRS)) * stepp / tauw;

            GE_ampa[i] += stepp * (-GE_ampa[i] / tau_decay_ampa);
            GE_nmda[i] += stepp * (-GE_nmda[i] / tau_decay_nmda);
            GE_gabaa[i] += stepp * (-GE_gabaa[i] / tau_decay_gabaa);

            GE_gabab[i] += stepp * (-GE_gabab[i] / tau_decay_gabaB + alpha_gabaB * QE_gabab[i]);
            QE_gabab[i] += stepp * (-QE_gabab[i] / tau_rise_gabaB);
        }

        /*INHIBITORY*/

        for (i = dimmexc; i < dim; i++)
        {

            randnumb = rand() / (RAND_MAX + 1.0);

            if (randnumb < stepp * inputrateonexc * pconn * dimmexc)
            {
                GI_ampa[i - dimmexc] += g_ampa;
            }

            randnumb = rand() / (RAND_MAX + 1.0);

            if (randnumb < stepp * inputrateoninh * pconn * diminh)
            {
                GI_gabaa[i - dimmexc] += g_gabaa;
            }

            v[i] += (glFS * (ElFS - v[i]) / CmFS + glFS * deltaFS * exp(((v[i] - thr) / (deltaFS))) / CmFS + GI_ampa[i - dimmexc] * (Ee - v[i]) / CmFS + GI_gabaa[i - dimmexc] * (Ei - v[i]) / CmFS + x_nmda(v[i]) * g_nmda * GI_nmda[i - dimmexc] * (Ee - v[i]) / CmFS + g_gabaB * GI_gabab[i - dimmexc] * (EgabaB - v[i]) / CmFS) * stepp * tetaaa(timet - tolsp[i] - Trefr);

            GI_ampa[i - dimmexc] += stepp * (-GI_ampa[i - dimmexc] / tau_decay_ampa);
            GI_nmda[i - dimmexc] += stepp * (-GI_nmda[i - dimmexc] / tau_decay_nmda);
            GI_gabaa[i - dimmexc] += stepp * (-GI_gabaa[i - dimmexc] / tau_decay_gabaa);

            GI_gabab[i] += stepp * (-GI_gabab[i] / tau_decay_gabaB + alpha_gabaB * QI_gabab[i]);
            QI_gabab[i] += stepp * (-QI_gabab[i] / tau_rise_gabaB);
        }

        for (i = 0; i < dim; i++)
        {

            if ((timet - spike_flag[i] - delay) >= 0)
            {

                spike_flag[i] = 1000000;

                for (j = 0; j < dimmexc; j++)
                {

                    if (i < ne)
                    {
                        GE_ampa[j] += g_ampa * (gij_weights[j][i] * gij_connections[j][i]);
                        GE_nmda[j] += (gij_weights[j][i] * gij_connections[j][i]);
                    }
                    else
                    {
                        GE_gabaa[j] += (g_gabaa * gij_weights[j][i] * gij_connections[j][i]);

                        QE_gabab[j] += deltaq_gabaB * gij_weights[j][i] * gij_connections[j][i];
                    }
                }

                for (j = 0; j < diminh; j++)
                {

                    if (i < ne)
                    {
                        GI_ampa[j] += g_ampa * (gij_weights[j + dimmexc][i] * gij_connections[j + dimmexc][i]);
                        GI_nmda[j] += (gij_weights[j + dimmexc][i] * gij_connections[j + dimmexc][i]);
                    }
                    else
                    {
                        GI_gabaa[j] += (g_gabaa * gij_weights[j + dimmexc][i] * gij_connections[j + dimmexc][i]);
                        QI_gabab[j] += deltaq_gabaB * gij_weights[j + dimmexc][i] * gij_connections[j + dimmexc][i];
                    }
                }
            }
        }

        if (sqrt(dopamine * dopamine) > 5.e-1)
        {

            for (i = 0; i < dimmexc; i++)
            {

                for (j = 0; j < dimmexc; j++)
                {

                    eligibility[i][j] = eligibility[i][j] * exp((-timet + lastupdate[i][j]) / tauc);
                    lastupdate[i][j] = timet;

                    if (gij_weights[i][j] < wm)
                    {
                        gij_weights[i][j] += 100 * (dopamine)*eligibility[i][j] * stepp;
                    }
                    if (gij_weights[i][j] < wmin)
                    {
                        gij_weights[i][j] = wmin;
                    }
                }
            }
        }

        for (i = 0; i < dimmexc; i++)
        {

            vold[i] = v[i];
            if (v[i] > vspike)
            {

                rexc += 1;

                tolsp[i] = timet;
                spike_flag[i] = timet;

                fprintf(rasterexc, "%lf    %d\n", timet, i);

                ratebinn += 1;
                v[i] = vres;
                ww[i] += bRS;

                /*PLASTICITY*/

                for (j = 0; j < dimmexc; j++)
                {               // excitatory neurons
                    if (i != j) // excitatory neurons

                    {
                        deltaij = timet - tolsp[j];

                        eligibility[i][j] = eligibility[i][j] * exp((-timet + lastupdate[i][j]) / tauc);

                        lastupdate[i][j] = timet;

                        eligibility[i][j] += pp * tetaaa(em - eligibility[i][j]) * exp(-deltaij / taup);

                        eligibility[j][i] -= dd * (eligibility[j][i]) * exp(-deltaij / taum);
                    }
                }
            }
        }

        for (i = dimmexc; i < dim; i++)
        {

            vold[i] = v[i];
            if (v[i] > vspike)
            {

                rinh += 1;

                tolsp[i] = timet;
                spike_flag[i] = timet;
                fprintf(rasterinh, "%lf    %d\n", timet, i);

                ratebinn += 1;
                v[i] = vres;
            }
        }
    }

    for (i = 0; i < dimmexc; i++)
    {
        for (j = 0; j < dimmexc; j++)
        {

            fprintf(conn_f, "%d    %d  %lf\n", i, j, gij_weights[i][j]);
        }
    }
}




double tetaaa(double juy)
{

    if (juy >= 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
double heaviside(double x)
{
    if (x >= 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
double doubgauss(double t, double t0, double T1, double T2, double amp)
{
    return amp * (exp(-((t - t0) * (t - t0)) / (2 * T1 * T1)) * inputextern(-(t - t0)) + exp(-((t - t0) * (t - t0)) / (2 * T2 * T2)) * heaviside(t - t0));
}
double rand_lognormal(double mean, double stddev)
{
    double u1 = rand() / (double)RAND_MAX;
    double u2 = rand() / (double)RAND_MAX;

    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    double x = exp(mean + stddev * z);

    return x;
}
double x_nmda(double juy)
{

    return 1. / (1 + 1.5 * exp(-0.062 * juy) / 3.57);
}
