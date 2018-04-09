#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"

const float SPD = 3e11; //light speed (mm/s)
// const float eps = 1e-7;
// const float FLT_MAX = 1e13;
const int NB_THREADS = 1024;
const int NB_BLOCKS = 36;

const float realmin = 1.0e-13; //DBL_MIN
const float realmax = 1.0e13;  //DBL_MAX

struct TOFInfo
{
	const float time_resolution;
	const float tof_bin;
};
struct EventsInfo
{
	EventsInfo(int nb_events, int event_size)
		: nb_events(nb_events), event_size(event_size) {}
	const long long nb_events;
	const int event_size;
};
struct ImageInfo
{
	const int grid[3];
	const float center[3];
	const float size[3];
};

__global__ void TestFloat3Kernel()
{
	float3 v = make_float3(1., 2., 3.);
	printf("%f %f %f\n", v.x, v.y, v.z);
}

void TestFloat3(const TOFInfo &c)
{
	std::cout << c.time_resolution << std::endl;
	TestFloat3Kernel<<<32, 32>>>();
}
/*

__device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 operator/(const float3 &a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}
__device__ float3 operator*(const float3 &a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float3 operator+(const float3 &a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
__device__ float3 operator-(const float3 &a, float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}
__device__ float3 min(const float3 &a, const float3 &b)
{
	return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
__device__ float3 max(const float3 &a, const float3 &b)
{
	return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
__device__ float min(const float3 &a)
{
	return min(min(a.x, a.y), a.z);
}
__device__ float operator()(const float3 &a, int i)
{
	if (i == 0)
		return a.x;
	if (i == 1)
		return a.y;
	if (i == 2)
		return a.z;
}

__device__ void to_list(const float3 a, float[] b)
{
	b[0] = a.x;
	b[1] = a.y;
	b[2] = a.z;
}

*/

__device__ void cal_crs(const float *lor, const int *grid, const float *orgin, const float *size,
						float *alpha, float *alphau, int *index, float &dconv, int *u, int &Np)
/* alpha[0]: alphax; alpha[1]: alphay; alpha[2]: alphaz; alpha[3]: alphac;
   alphau[0]: alphaxu; alphau[1]: alphayu; alphau[2]:alphazu;
   index[0]: xindex; index[1]: yindex; index[2]: zindex;
   u[0]:iu; u[1]:ju; u[2]:ku;

*/
{
	float alphax0, alphaxn, alphay0, alphayn, alphaz0, alphazn;
	float alphaxmin, alphaxmax, alphaymin, alphaymax, alphazmin, alphazmax;
	float alphamin, alphamax, alphaavg;

	float alphatemp;
	float phyx_alphamin, phyx_alphamax, phyy_alphamin, phyy_alphamax, phyz_alphamin, phyz_alphamax;

	float p1x, p1y, p1z, p2x, p2y, p2z, pdx, pdy, pdz;

	int i_f, i_l, j_f, j_l, k_f, k_l, i_min, i_max, j_min, j_max, k_min, k_max;
	int Nx, Ny, Nz;

    //printf("lor[0]=%f lor[1]=%f lor[2]=%f lor[3]=%f lor[4]=%f lor[5]=%f\n", lor[0], lor[1],lor[2], lor[3],lor[4],lor[5]);
	p1x = lor[0]; // sou_p[0];
	p2x = lor[3]; // end_p[0];
	pdx = p2x - p1x;

	p1y = lor[1]; // sou_p[1];
	p2y = lor[4]; // end_p[1];
	pdy = p2y - p1y;

	p1z = lor[2]; // sou_p[2];
	p2z = lor[5]; // end_p[2];
	pdz = p2z - p1z;

	//printf("p1x=%f p2x=%f pdx=%f p1y=%f p2y=%f pdy=%f p1z=%f p2z=%f pdz=%f\n", p1x,p2x,pdx,p1y,p2y,pdy,p1z,p2z,pdz);

	Nx = grid[2] + 1;
	Ny = grid[1] + 1;
	Nz = grid[0] + 1;

	alphax0 = (orgin[2] - p1x) / (pdx + realmin);
	alphaxn = (orgin[2] + (Nx - 1) * size[2] - p1x) / (pdx + realmin);

	alphay0 = (orgin[1] - p1y) / (pdy + realmin);
	alphayn = (orgin[1] + (Ny - 1) * size[1] - p1y) / (pdy + realmin);

	alphaz0 = (orgin[0] - p1z) / (pdz + realmin);
	alphazn = (orgin[0] + (Nz - 1) * size[0] - p1z) / (pdz + realmin);

	alphaxmin = min(alphax0, alphaxn);
	alphaxmax = max(alphax0, alphaxn);
	alphaymin = min(alphay0, alphayn);
	alphaymax = max(alphay0, alphayn);
	alphazmin = min(alphaz0, alphazn);
	alphazmax = max(alphaz0, alphazn);

	alphatemp = max(alphaxmin, alphaymin);
	alphamin = max(alphatemp, alphazmin);

	alphatemp = min(alphaxmax, alphaymax);
	alphamax = min(alphatemp, alphazmax);

	if (alphamin < alphamax)
	{
		phyx_alphamin = (p1x + alphamin * pdx - orgin[2]) / size[2];
		phyx_alphamax = (p1x + alphamax * pdx - orgin[2]) / size[2];

		phyy_alphamin = (p1y + alphamin * pdy - orgin[1]) / size[1];
		phyy_alphamax = (p1y + alphamax * pdy - orgin[1]) / size[1];

		phyz_alphamin = (p1z + alphamin * pdz - orgin[0]) / size[0];
		phyz_alphamax = (p1z + alphamax * pdz - orgin[0]) / size[0];

		if (p1x < p2x)
		{
			if (alphamin == alphaxmin)
				i_f = 1;
			else
				i_f = ceil(phyx_alphamin);
			if (alphamax == alphaxmax)
				i_l = Nx - 1;
			else
				i_l = floor(phyx_alphamax);
			u[2] = 1;
			alpha[2] = (orgin[2] + i_f * size[2] - p1x) / pdx;
		}
		else if (p1x > p2x)
		{
			if (alphamin == alphaxmin)
				i_f = Nx - 2;
			else
				i_f = floor(phyx_alphamin);
			if (alphamax == alphaxmax)
				i_l = 0;
			else
				i_l = ceil(phyx_alphamax);
			u[2] = -1;
			alpha[2] = (orgin[2] + i_f * size[2] - p1x) / pdx;
		}
		else
		{
			i_f = int(phyx_alphamin);
			i_l = int(phyx_alphamax);
			u[2] = 0;
			alpha[2] = realmax;
		}

		if (p1y < p2y)
		{
			if (alphamin == alphaymin)
				j_f = 1;
			else
				j_f = ceil(phyy_alphamin);
			if (alphamax == alphaymax)
				j_l = Ny - 1;
			else
				j_l = floor(phyy_alphamax);
			u[1] = 1;
			alpha[1] = (orgin[1] + j_f * size[1] - p1y) / pdy;
		}
		else if (p1y > p2y)
		{
			if (alphamin == alphaymin)
				j_f = Ny - 2;
			else
				j_f = floor(phyy_alphamin);
			if (alphamax == alphaymax)
				j_l = 0;
			else
				j_l = ceil(phyy_alphamax);
			u[1] = -1;
			alpha[1] = (orgin[1] + j_f * size[1] - p1y) / pdy;
		}
		else
		{
			j_f = int(phyy_alphamin);
			j_l = int(phyy_alphamax);
			u[1] = 0;
			alpha[1] = realmax;
		}

		if (p1z < p2z)
		{
			if (alphamin == alphazmin)
				k_f = 1;
			else
				k_f = ceil(phyz_alphamin);
			if (alphamax == alphazmax)
				k_l = Nz - 1;
			else
				k_l = floor(phyz_alphamax);
			u[0] = 1;
			alpha[0] = (orgin[0] + k_f * size[0] - p1z) / pdz;
		}
		else if (p1z > p2z)
		{
			if (alphamin == alphazmin)
				k_f = Nz - 2;
			else
				k_f = floor(phyz_alphamin);
			if (alphamax == alphazmax)
				k_l = 0;
			else
				k_l = ceil(phyz_alphamax);
			u[0] = -1;
			alpha[0] = (orgin[0] + k_f * size[0] - p1z) / pdz;
		}
		else
		{
			k_f = int(phyz_alphamin);
			k_l = int(phyz_alphamax);
			u[0] = 0;
			alpha[0] = realmax;
		}

		i_min = min(i_f, i_l);
		i_max = max(i_f, i_l);
		j_min = min(j_f, j_l);
		j_max = max(j_f, j_l);
		k_min = min(k_f, k_l);
		k_max = max(k_f, k_l);
		
		Np = (i_max - i_min+1) + (j_max - j_min+1) + (k_max - k_min+1);
		//printf("np: %d, imax %d, imin %d, jmax %d jmin %d k_max %d kmin %d\n", Np, i_max, i_min, j_max, j_min, k_max, k_min);
		alphatemp = min(alpha[2], alpha[1]);
		alphaavg = (min(alphatemp, alpha[0]) + alphamin) / 2;

		index[2] = int(((p1x + alphaavg * pdx) - orgin[2]) / size[2]);
		index[1] = int(((p1y + alphaavg * pdy) - orgin[1]) / size[1]);
		index[0] = int(((p1z + alphaavg * pdz) - orgin[0]) / size[0]);

/*
		printf("pdx=%f\n", pdx);
		printf("pdy=%f\n", pdy);
		printf("pdz=%f\n", pdz);
		printf("p1x=%f\n", p1x);
		printf("p1y=%f\n", p1y);
		printf("p1z=%f\n", p1z);
		printf("orgin[0]=%f\n", orgin[0]);
		printf("orgin[1]=%f\n", orgin[1]);
		printf("orgin[2]=%f\n", orgin[2]);

		printf("size[0]=%f\n", size[0]);
		printf("size[1]=%f\n", size[1]);
		printf("size[2]=%f\n", size[2]);

		printf("index[0]=%d\n", index[0]);
		printf("index[1]=%d\n", index[1]);
		printf("index[2]=%d\n", index[2]);
		*/
		alphau[2] = size[2] / (fabs(pdx) + realmin);
		alphau[1] = size[1] / (fabs(pdy) + realmin);
		alphau[0] = size[0] / (fabs(pdz) + realmin);

		alpha[3] = alphamin;
	//	printf("p1x=%f p2x=%f pdx=%f p1y=%f p2y=%f pdy=%f p1z=%f p2z=%f pdz=%f\n", p1x, p2x, pdx, p1y, p2y, pdy, p1z, p2z, pdz);
		dconv = sqrt(pdx * pdx + pdy * pdy + pdz * pdz);
	//	printf("alpha[0]=%f alpha[1]=%f alpha[2]=%f alpha[3]=%f\n", alpha[0], alpha[1], alpha[2], alpha[3]);

	}
}

__device__ void cal_weight(float *alpha, float *alphau, int *index, int *u, float dconv, float DETLA_TOF,
						   float TOF_tr, float TOF_dif_time, const int *grid, int &tmp_index, float &tmp_data)
/* alpha[0]: alphax; alpha[1]: alphay; alpha[2]: alphaz; alpha[3]: alphac;
   alphau[0]: alphaxu; alphau[1]: alphayu; alphau[2]:alphazu;
   index[0]: xindex; index[1]: yindex; index[2]: zindex;
   u[0]:iu; u[1]:ju; u[2]:ku;

*/
{
	float TOF, tof_t;
	float c, b;

	//printf("alpha[0]=%f alpha[1]=%f alpha[2]=%f alpha[3]=%f\n", alpha[0], alpha[1], alpha[2], alpha[3]);

	//	printf("Entered:");
		if ((alpha[2] < alpha[1]) && (alpha[2] < alpha[0]))
		{
		//	printf("alpha[0]=%f alpha[1]=%f alpha[2]=%f alpha[3]=%f\n", alpha[0], alpha[1], alpha[2], alpha[3]);

	//		printf("Branch 0\n");
				tof_t = ((alpha[2] + alpha[3] - 1.0) * dconv / 2.0) / DETLA_TOF;
				c = (TOF_tr * SPD / 2. / 2. / sqrt(2 * log(2.0))) / DETLA_TOF;
				b = (TOF_dif_time * SPD) / DETLA_TOF;
			//	printf("DETLA_TOF=%f\n", DETLA_TOF);
			//	printf("((alpha[2] + alpha[3]) * dconv / 2.0 - dconv / 2.0)=%f\n", ((alpha[2] + alpha[3]) * dconv / 2.0 - dconv / 2.0));
			//	printf("alpha[0]=%f alpha[1]=%f alpha[2]=%f alpha[3]=%f\n", alpha[0], alpha[1], alpha[2], alpha[3]);

		//		printf("alpha[2] + alpha[3]=%f\n", (alpha[2] + alpha[3]-1.0));

			//	printf("c=%f\n", c);
		//		printf("b=%f\n", b);
			//	printf("tof_t=%f\n", tof_t);

				if ((tof_t > (b - 3 * c)) & (tof_t < (b + 3 * c)))
				{
					TOF = exp(-((tof_t - b) * (tof_t - b)) / (2 * c * c));
				}

				else
				{
					TOF = 0.0;
				}


			
			//tmp_index = index[2] * grid[1] * grid[0] + index[0] * grid[1] + index[1];
	                tmp_index = index[0]* grid[1]*grid[2]+index[1]*grid[2]+index[2];
					tmp_data = (alpha[2] - alpha[3]) * dconv * TOF;
                 //       printf("index[0]=%d\n",index[0]);
                  //     printf("index[1]=%d\n",index[1]);
                  //     printf("index[2]=%d\n",index[2]);
             //           printf("tmp_index=%d\n",tmp_index);
          //                printf("alpha[2]-alpha[3]=%f\n",alpha[2]-alpha[3]);
          //              printf("dconv=%f\n",dconv);
          //              printf("TOF=%f\n",TOF) ; 




			index[2] = index[2] + u[2];
			alpha[3] = alpha[2];
			alpha[2] = alpha[2] + alphau[2];
                        
		}
		else if (alpha[1] < alpha[0])
		{			
			
				tof_t = ((alpha[1] + alpha[3] - 1.0 ) * dconv / 2.0) / DETLA_TOF;
				c = (TOF_tr * SPD / 2. / 2. / sqrt(2 * log(2.0))) / DETLA_TOF;
				b = (TOF_dif_time * SPD) / DETLA_TOF;

				if ((tof_t > (b - 3 * c)) & (tof_t < (b + 3 * c)))
					TOF = exp(-((tof_t - b) * (tof_t - b)) / (2 * c * c));
				else
					TOF = 0.;

		//	tmp_index = index[2] * grid[1] * grid[0] + index[0] * grid[1] + index[1];
                        tmp_index = index[0]* grid[1]*grid[2]+index[1]*grid[2]+index[2];

						tmp_data = (alpha[1] - alpha[3]) * dconv *TOF;

      //                 printf("index[0]=%d\n",index[0]);
      //                  printf("index[1]=%d\n",index[1]);
      //                  printf("index[2]=%d\n",index[2]);
       //                 printf("tmp_index=%d\n",tmp_index);

      //                  printf("alpha[1]-alpha[3]=%f\n",alpha[1]-alpha[3]);
      //                 printf("dconv=%f\n",dconv);
       //                 printf("TOF=%f\n",TOF) ;



			index[1] = index[1] + u[1];
			alpha[3] = alpha[1];
			alpha[1] = alpha[1] + alphau[1];
 

		}

		else
		{			

				tof_t = ((alpha[0] + alpha[3] - 1.0) * dconv / 2.0) / DETLA_TOF;
				c = (TOF_tr * SPD / 2. / 2. / sqrt(2 * log(2.0))) / DETLA_TOF;
				b = (TOF_dif_time * SPD) / DETLA_TOF;

				if ((tof_t > (b - 3 * c)) & (tof_t < (b + 3 * c)))
					TOF = exp(-((tof_t - b) * (tof_t - b)) / (2 * c * c));
				else
					TOF = 0.;
			
//			tmp_index = index[2] * grid[1] * grid[0] + index[0] * grid[1] + index[1];
		                        tmp_index = index[0]* grid[1]*grid[2]+index[1]*grid[2]+index[2];

								tmp_data = (alpha[0] - alpha[3]) * dconv *TOF;
           //             printf("alpha[0]-alpha[3]=%f\n",alpha[0]-alpha[3]);
           //             printf("dconv=%f\n",dconv);
           //             printf("TOF=%f\n",TOF) ;
           //            printf("index[0]=%d\n",index[0]);
            //           printf("index[1]=%d\n",index[1]);
            //            printf("index[2]=%d\n",index[2]);
    //                    printf("tmp_index=%d\n",tmp_index);

			index[0] = index[0] + u[0];
			alpha[3] = alpha[0];
			alpha[0] = alpha[0] + alphau[0];
 

		}
	
}

__device__ void ProjectionOneEvent(const float *lor, const float *tofinfo, const int *grid, const float *orgin, const float *size, const float *image, float *lor_value)
{
	float DETLA_TOF = tofinfo[0] * SPD; // TOF_td * SPD; // the TOF.Detla mostly lower than 30e-12 (usually I set it to 10e-12)
	float TOF_tr = tofinfo[1];			//TOF resolution for PET system (unit: s)
	float TOF_dif_time = lor[6];

	int i;
	float dconv;
	float alpha[4];
	float alphau[3];
	int u[3];
	int index[3];
	int Np = 0;

	int tmp_index=0;
	float tmp_data=0.;
	
	cal_crs(lor, grid, orgin, size, alpha, alphau, index, dconv, u, Np);
	for (i = 0; i < Np; i++)
	{
		// printf("flag %d\n", i);
		if ((index[0] >= 0) && (index[0] <= (grid[0] - 1)) && (index[1] >= 0) && (index[1] <= (grid[1] - 1)) && (index[2] >= 0) && (index[2] <= (grid[2] - 1)))
		{
			cal_weight(alpha, alphau, index, u, dconv, DETLA_TOF, TOF_tr, TOF_dif_time, grid, tmp_index, tmp_data);
			printf("index[0]=%d, index[1]=%d, index[2]=%d\n", index[0], index[1], index[2]);
			printf("tid %d, tdata %f\n", tmp_index, tmp_data);
			lor_value[0] += image[tmp_index] * tmp_data;
		}
	}
}
__device__ void BackProjectionOneEvent(const float *lor, const float *tofinfo, const int *grid, const float *orgin, const float *size, float *image, const float *lor_value)
{
	float DETLA_TOF = tofinfo[0] * SPD; // TOF_td * SPD; // the TOF.Detla mostly lower than 30e-12 (usually I set it to 10e-12)
	float TOF_tr = tofinfo[1];			//TOF resolution for PET system (unit: s)
	float TOF_dif_time = lor[6];

	int i;
	float dconv;
	float alpha[4];
	float alphau[3];
	int u[3];
	int index[3];
	int Np=0;

	int tmp_index=0;
	float tmp_data=0.;
      


	cal_crs(lor, grid, orgin, size, alpha, alphau, index, dconv, u, Np);
        
    //    printf("Np=%d\n",Np);
 //       printf("lor_value[0]=%f\n",lor_value[0]);  
	
	for (i = 0; i < Np; i++)
	{
		if ((index[0] >= 0) && (index[0] <= (grid[0] - 1)) && (index[1] >= 0) && (index[1] <= (grid[1] - 1)) && (index[2] >= 0) && (index[2] <= (grid[2] - 1)))
		{
		//	printf("i=%d\n", i);
			cal_weight(alpha, alphau, index, u, dconv, DETLA_TOF, TOF_tr, TOF_dif_time, grid, tmp_index, tmp_data);
			
			//printf("lor_value[0]=%f\n", lor_value[0]);
			//image[tmp_index] += lor_value[0] * tmp_data;
			atomicAdd(image + tmp_index, lor_value[0] + tmp_data);
	//		if ((lor_value[0]-1.0)>realmin)

//			printf("tmp_data=%f\n",tmp_data);
                 //       printf("grid[0]=%d\n",grid[0]);
                 //       printf("grid[1]=%d\n",grid[1]);
		//	printf("grid[2]=%d\n",grid[2]);
              //          printf("image[%d]=%f\n ", tmp_index, image[tmp_index]);
		}
	}
	
}

__device__ void PrepareInputs(const EventsInfo events_info,
						 const ImageInfo image_info,
						 const TOFInfo tof_info,
						 int gird[],
						 float origin[],
						 float size_[],
						 float tof_info_[])
{
	tof_info_[0] = tof_info.tof_bin;
	tof_info_[1] = tof_info.time_resolution;

	for (int i = 0; i < 3; ++i)
	{
		gird[i] = image_info.grid[i];
		origin[i] = image_info.center[i];// -image_info.size[i];
		size_[i] = image_info.size[i];
	}
}

__global__ void ProjectionKernel(const float *lor,
								 const EventsInfo events_info,
								 const float *image,
								 const ImageInfo image_info,
								 const TOFInfo tof_info,
								 float *result)
{
	const int step = gridDim.x * blockDim.x;
	float tof_info_[2], origin_[3], size_[3];
	int grid_[3];
	PrepareInputs(events_info, image_info, tof_info, grid_, origin_, size_, tof_info_);
	// printf("ok\n");	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < events_info.nb_events; i += step)
	{		
		ProjectionOneEvent(lor + i * 7,
						   tof_info_, grid_, origin_, size_, image,
						   result);
	}
}
__global__ void BackProjectionKernel(const float *lor,
									 const float *lor_values,
									 const EventsInfo events_info,
									 const ImageInfo image_info,
									 const TOFInfo tof_info,
									 float *result)
{
	const int step = gridDim.x * blockDim.x;
	float tof_info_[2], origin_[3], size_[3];
	int grid_[3];
	PrepareInputs(events_info, image_info, tof_info, grid_, origin_, size_, tof_info_);
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < events_info.nb_events; i += step)
	{
		//printf("tid %d\t, lor value: %f\n", i, *(lor_values + i));
		BackProjectionOneEvent(lor + i * events_info.event_size,
							   tof_info_, grid_, origin_, size_,
							   result, lor_values + i);
	}
}
void ProjectionKernelLauncher(const float *lor,
							  const EventsInfo events_info,
							  const float *image,
							  const ImageInfo image_info,
							  const TOFInfo tof_info,
							  float *result)
{
	ProjectionKernel<<<NB_BLOCKS, NB_THREADS>>>(lor, events_info, image, image_info, tof_info, result);
}
void BackProjectionKernelLauncher(const float *lor,
								  const float *lor_values,
								  const EventsInfo events_info,
								  const ImageInfo image_info,
								  const TOFInfo tof_info,
								  float *result)
{
	BackProjectionKernel<<<NB_BLOCKS, NB_THREADS>>>(lor, lor_values, events_info, image_info, tof_info, result);


}
#endif
