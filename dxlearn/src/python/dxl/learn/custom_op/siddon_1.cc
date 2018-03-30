

#define realmin 1.0e-13 //DBL_MIN
#define realmax 1.0e13 //DBL_MAX
#define SPD 3E10  //light speed (cm/s)
//#include "/home/chengaoyu/code/C++/BBSLMIRP_QT/PETSystem/petapplication.h"




	void CalculateFactor(float *lor, float *tofinfo, int *grid, float *orgin, float *size,  float *image, float *lor_value, int flag)
		/*
		flag==0: project function
			input parameter: 
							lor: lor[0], lor[1], lor[2] is X Y Z coordinate of the first xtal, lor[3], lor[4], lor[5] is X Y Z coordinate of the second xtal, lor[6] is different time
							tofinfo: tofinfo[0] is TOF bin size (unit: ps), tofinfo[1] is TOF resolution (unit: ps)
							grid:   grid[0], grid[1], grid[2] is the voxels number of image in X Y Z axis
							orgin:  orgin[0], orgin[1], orgin[2] is the left-lower corner voxels number of image in X Y Z axis
							size:   size[0], size[1], size[2] is the voxel size of image in X Y Z axis
							image:  image to be projected
			output parameter:
							lor_value: projection
		
		flag==1: back-project function
			input parameter:
							lor: lor[0], lor[1], lor[2] is X Y Z coordinate of the first xtal, lor[3], lor[4], lor[5] is X Y Z coordinate of the second xtal, lor[6] is different time
							tofinfo: tofinfo[0] is TOF bin size (unit: ps), tofinfo[1] is TOF resolution (unit: ps)
							grid:   grid[0], grid[1], grid[2] is the voxels number of image in X Y Z axis
							orgin:  orgin[0], orgin[1], orgin[2] is the left-lower corner voxels number of image in X Y Z axis
							size:   size[0], size[1], size[2] is the voxel size of image in X Y Z axis
							lor_value:  projection to be back-projected
			output parameter:
							image: back-projected image
		*/
	{  


		


		float DETLA_TOF = tofinfo[0]*SPD;// TOF_td * SPD; // the TOF.Detla mostly lower than 30e-12 (usually I set it to 10e-12)
		float TOF_tr = tofinfo[1]; //TOF resolution for PET system (unit: s)
		float alphax0, alphaxn, alphay0, alphayn, alphaz0, alphazn;
		float alphaxmin, alphaxmax, alphaymin, alphaymax, alphazmin, alphazmax, alphac;
		float alphamin, alphamax, alphaavg, alphax, alphay, alphaz, alphaxu, alphayu, alphazu;
		float alphatemp;
		float phyx_alphamin, phyx_alphamax, phyy_alphamin, phyy_alphamax, phyz_alphamin, phyz_alphamax;

		float p1x, p1y, p1z, p2x, p2y, p2z, pdx, pdy, pdz;
		float dconv;

		int i_f, i_l, j_f, j_l, k_f, k_l, i_min, i_max, j_min, j_max, k_min, k_max, iu, ju, ku;
		int Nx, Ny, Nz, Np;
		int i, xindex, yindex, zindex;

		float b, c;
		float TOF;
		float tof_t;

		float TOF_dif_time = lor[6];
		int tmp_index;
		float tmp_data;



		p1x = lor[0];// sou_p[0];
		p2x = lor[3];// end_p[0];
		pdx = p2x - p1x;

		p1y = lor[1];// sou_p[1];
		p2y = lor[4];// end_p[1];
		pdy = p2y - p1y;

		p1z = lor[2];// sou_p[2];
		p2z = lor[5];// end_p[2];
		pdz = p2z - p1z;

		Nx = grid[0] + 1;
		Ny = grid[1] + 1;
		Nz = grid[2] + 1;

		alphax0 = (orgin[0] - p1x) / (pdx + realmin);
		alphaxn = (orgin[0] + (Nx - 1)*size[0] - p1x) / (pdx + realmin);

		alphay0 = (orgin[1] - p1y) / (pdy + realmin);
		alphayn = (orgin[1] + (Ny - 1)*size[1] - p1y) / (pdy + realmin);

		alphaz0 = (orgin[2] - p1z) / (pdz + realmin);
		alphazn = (orgin[2] + (Nz - 1)*size[2] - p1z) / (pdz + realmin);



		alphaxmin = min(alphax0, alphaxn); alphaxmax = max(alphax0, alphaxn);
		alphaymin = min(alphay0, alphayn); alphaymax = max(alphay0, alphayn);
		alphazmin = min(alphaz0, alphazn); alphazmax = max(alphaz0, alphazn);

		alphatemp = max(alphaxmin, alphaymin);
		alphamin = max(alphatemp, alphazmin);

		alphatemp = min(alphaxmax, alphaymax);
		alphamax = min(alphatemp, alphazmax);

		if (alphamin < alphamax)
		{
			phyx_alphamin = (p1x + alphamin * pdx - orgin[0]) / size[0];
			phyx_alphamax = (p1x + alphamax * pdx - orgin[0]) / size[0];

			phyy_alphamin = (p1y + alphamin * pdy - orgin[1]) / size[1];
			phyy_alphamax = (p1y + alphamax * pdy - orgin[1]) / size[1];

			phyz_alphamin = (p1z + alphamin * pdz - orgin[2]) / size[2];
			phyz_alphamax = (p1z + alphamax * pdz - orgin[2]) / size[2];


			if (p1x < p2x)
			{
				if (alphamin == alphaxmin)    i_f = 1;
				else                          i_f = ceil(phyx_alphamin);
				if (alphamax == alphaxmax)    i_l = Nx - 1;
				else                          i_l = floor(phyx_alphamax);
				iu = 1;
				alphax = (orgin[0] + i_f * size[0] - p1x) / pdx;
			}
			else if (p1x > p2x)
			{
				if (alphamin == alphaxmin)    i_f = Nx - 2;
				else                          i_f = floor(phyx_alphamin);
				if (alphamax == alphaxmax)    i_l = 0;
				else                          i_l = ceil(phyx_alphamax);
				iu = -1;
				alphax = (orgin[0] + i_f * size[0] - p1x) / pdx;
			}
			else
			{
				i_f = int(phyx_alphamin);
				i_l = int(phyx_alphamax);
				iu = 0;
				alphax = realmax;
			}

			if (p1y < p2y)
			{
				if (alphamin == alphaymin)    j_f = 1;
				else                          j_f = ceil(phyy_alphamin);
				if (alphamax == alphaymax)    j_l = Ny - 1;
				else                          j_l = floor(phyy_alphamax);
				ju = 1;
				alphay = (orgin[1] + j_f * size[1] - p1y) / pdy;
			}
			else if (p1y > p2y)
			{
				if (alphamin == alphaymin)    j_f = Ny - 2;
				else                          j_f = floor(phyy_alphamin);
				if (alphamax == alphaymax)    j_l = 0;
				else                          j_l = ceil(phyy_alphamax);
				ju = -1;
				alphay = (orgin[1] + j_f * size[1] - p1y) / pdy;
			}
			else
			{
				j_f = int(phyy_alphamin);
				j_l = int(phyy_alphamax);
				ju = 0;
				alphay = realmax;
			}


			if (p1z < p2z)
			{
				if (alphamin == alphazmin)    k_f = 1;
				else                          k_f = ceil(phyz_alphamin);
				if (alphamax == alphazmax)    k_l = Nz - 1;
				else                          k_l = floor(phyz_alphamax);
				ku = 1;
				alphaz = (orgin[2] + k_f * size[2] - p1z) / pdz;
			}
			else if (p1z > p2z)
			{
				if (alphamin == alphazmin)    k_f = Nz - 2;
				else                          k_f = floor(phyz_alphamin);
				if (alphamax == alphazmax)    k_l = 0;
				else                          k_l = ceil(phyz_alphamax);
				ku = -1;
				alphaz = (orgin[2] + k_f * size[2] - p1z) / pdz;
			}
			else
			{
				k_f = int(phyz_alphamin);
				k_l = int(phyz_alphamax);
				ku = 0;
				alphaz = realmax;
			}

			i_min = min(i_f, i_l);
			i_max = max(i_f, i_l);
			j_min = min(j_f, j_l);
			j_max = max(j_f, j_l);
			k_min = min(k_f, k_l);
			k_max = max(k_f, k_l);


			Np = (i_max - i_min + 1) + (j_max - j_min + 1) + (k_max - k_min + 1);

			alphatemp = min(alphax, alphay);
			alphaavg = (min(alphatemp, alphaz) + alphamin) / 2;

			xindex = int(((p1x + alphaavg * pdx) - orgin[0]) / size[0]);
			yindex = int(((p1y + alphaavg * pdy) - orgin[1]) / size[1]);
			zindex = int(((p1z + alphaavg * pdz) - orgin[2]) / size[2]);

			alphaxu = size[0] / (fabs(pdx) + realmin);
			alphayu = size[1] / (fabs(pdy) + realmin);
			alphazu = size[2] / (fabs(pdz) + realmin);

			alphac = alphamin;
			dconv = sqrt(pdx*pdx + pdy * pdy + pdz * pdz);

			for (i = 0; i < Np; i++)
			{
				if (alphax < alphay && alphax < alphaz)
				{
					if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2) && zindex >= 0 && zindex <= (Nz - 2))
					{
						if (TOF_tr == 0)
						{
							TOF = 1.0;
						}
						else
						{
							tof_t = ((alphax + alphac)*dconv / 2.0 - dconv / 2.0) / DETLA_TOF;
							c = (TOF_tr*SPD / 2 / 2 / sqrt(2 * log(2.0))) / DETLA_TOF;
							b = (TOF_dif_time*SPD) / DETLA_TOF;

							if ((tof_t > (b - 3 * c))&(tof_t < (b + 3 * c)))
								TOF = exp(-((tof_t - b)*(tof_t - b)) / (2 * c*c));
							else
								TOF = 0.;

						}

						tmp_index = zindex * grid[1] * grid[0] + xindex * grid[1] + yindex;
						tmp_data  = (alphax - alphac)*dconv*TOF;


						xindex = xindex + iu;
						alphac = alphax;
						alphax = alphax + alphaxu;
					}
				}
				else if (alphay < alphaz)
				{
					if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2) && zindex >= 0 && zindex <= (Nz - 2))
					{
						if (TOF_tr == 0)
						{
							TOF = 1.0;
						}
						else
						{
							tof_t = ((alphay + alphac)*dconv / 2.0 - dconv / 2.0) / DETLA_TOF;
							c = (TOF_tr*SPD / 2 / 2 / sqrt(2 * log(2.0))) / DETLA_TOF;
							b = (TOF_dif_time*SPD) / DETLA_TOF;

							if ((tof_t > (b - 3 * c))&(tof_t < (b + 3 * c)))
								TOF = exp(-((tof_t - b)*(tof_t - b)) / (2 * c*c));
							else
								TOF = 0.;
						}
						tmp_index = zindex * grid[1] * grid[0] + xindex * grid[1] + yindex;
						tmp_data  = (alphay - alphac)*dconv*TOF;

						yindex = yindex + ju;
						alphac = alphay;
						alphay = alphay + alphayu;

					}
				}
				else
				{
					if (xindex >= 0 && xindex <= (Nx - 2) && yindex >= 0 && yindex <= (Ny - 2) && zindex >= 0 && zindex <= (Nz - 2))
					{
						if (TOF_tr == 0)
						{
							TOF = 1.0;
						}
						else
						{
							tof_t = ((alphaz + alphac)*dconv / 2.0 - dconv / 2.0) / DETLA_TOF;
							c = (TOF_tr*SPD / 2 / 2 / sqrt(2 * log(2.0))) / DETLA_TOF;
							b = (TOF_dif_time*SPD) / DETLA_TOF;

							if ((tof_t > (b - 3 * c))&(tof_t < (b + 3 * c)))
								TOF = exp(-((tof_t - b)*(tof_t - b)) / (2 * c*c));
							else
								TOF = 0.;
						}
						tmp_index = zindex * grid[1] * grid[0] + xindex * grid[1] + yindex;
						tmp_data  = (alphaz - alphac)*dconv*TOF;
						zindex = zindex + ku;
						alphac = alphaz;
						alphaz = alphaz + alphazu;
					}
				}

				if (flag == 0)  //project
				{
					 lor_value += image[tmp_index] * tmp_data;
				}
				else if (flag == 1)  //back-proj
				{
					image[tmp_index] = lor_value * tmp_data;
				}


			}

		}
	}


private:
	string model;

};

