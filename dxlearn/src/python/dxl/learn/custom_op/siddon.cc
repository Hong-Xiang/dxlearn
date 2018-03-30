#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <cmath>
#include <algorithm>

#define realmin 1.0e-13 //DBL_MIN
#define realmax 1.0e13 //DBL_MAX
#define SPD 3E10  //light speed (cm/s)
//#include "/home/chengaoyu/code/C++/BBSLMIRP_QT/PETSystem/petapplication.h"

using namespace tensorflow;
using namespace std;
//using namespace BBSLMIRP;

REGISTER_OP("Projection")
.Input("lors: float")
.Input("image: float")
.Input("tofinfo: float")
.Input("grid: int32")
.Input("orgin: float")
.Input("size: float")
.Input("atan_map: float")
.Output("lor_values: float")
.Attr("model: string")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
	c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 1));
	return Status::OK();
});

REGISTER_OP("Backprojection")
.Input("image: float")
.Input("lors:float")
.Input("lor_values: float")
.Input("tofinfo: float") // frist element: time bin (unit:s); second element: time resolution (unit: s)
.Input("grid: int32")
.Input("orgin: float")
.Input("size: float")
.Input("atan_map:float")
.Output("backpro_image: float")
.Attr("model: string")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
	//set the size of backpro_image the same as the input image.
	c->set_output(0, c->input(0));
	return Status::OK();
});


class Projection : public OpKernel
{
public:
	explicit Projection(OpKernelConstruction *context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("model", &model));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor & lors = context->input(0);
		const Tensor &image = context->input(1);
		const Tensor &tofinfo = context->input(2);
		const Tensor &grid = context->input(3);
		const Tensor &orgin = context->input(4);
		const Tensor &size = context->input(5);
		const Tensor &atan_map = context->input(6);

		// Create an output tensor
		Tensor *lor_value = NULL;

		//debug
		//std::cout<<"lor_shape:"<<lors.shape().dim_size(1)<<std::endl;

		// define the shape of output tensors.
		TensorShape out_shape;
		out_shape.AddDim(lors.shape().dim_size(0));
		//debug
		//std::cout<<"out_shape dim:"<<out_shape.dims()<<std::endl;
		//std::cout<<"out_shape dim0:"<<out_shape.dim_size(0)<<std::endl;

		//assign the size of output lor values.
		OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
			&lor_value));
		//set the initial lor_value to zero. 
		auto pv_flat = lor_value->flat<float>();
		for (int i = 0; i< pv_flat.size(); i++) {
			pv_flat(i) = 0;
		}

		projection(lors, lor_value, tofinfo, grid, orgin, size, image, atan_map);
	}

	void projection(const Tensor &events, Tensor* lor_value, 
		const Tensor &tofinfo, const Tensor &grid, const Tensor &orgin, const Tensor &size,
		const Tensor &image, const Tensor &atan_map)
	{
		
	    std::vector<int> tmp_index;
	    std::vector<float> tmp_data;



		auto image_flat = image.flat<float>();
		auto atan_map_flat = atan_map.flat<float>();
		auto lor_value_flat = lor_value->flat<float>();

		int ind, ind_vol;
		
		

		for (ind = 0; ind < events.dim_size(0); ind++)
		{
			auto event = events.Slice(ind, ind + 1);
			float atan_proj=0;
	 
			CalculateFactor(events, tofinfo, grid, orgin, size, tmp_index, tmp_data);

			
			for (ind_vol=0; ind_vol < tmp_index.size(); ind_vol++)
			{
				lor_value_flat(ind) += image_flat(tmp_index[ind_vol]) * tmp_data[ind_vol];
				
				atan_proj += atan_map_flat(tmp_index[ind_vol]) * tmp_data[ind_vol];
			}

			lor_value_flat(ind) *= exp(atan_proj);
			//printf("lor=%f\n", lor_value_flat(ind));
			
		}

	
	}
	void CalculateFactor(const Tensor &events, const Tensor &tofinfo, const Tensor &grid, const Tensor &orgin, const Tensor &size, std::vector<int> &tmp_index, std::vector<float> &tmp_data)
		/*input parametrer:
		coordinate in X,Y, Z axis of the start points of LOR: sou_p[3]
		coordinate in X,Y, Z axis of the start points of LOR: end_p[3]
		the struct to define the image volume( see the definition in header): VolPara
		the 3-D image volume: volume
		*/
	{
		auto tofinfo_flat = tofinfo.flat<float>();
		auto event_flat = events.flat<float>();
		auto grid_flat = grid.flat<int>();
		auto orgin_flat = orgin.flat<float>();
		auto size_flat = size.flat<float>();

		


		float DETLA_TOF = tofinfo_flat(0)*SPD;// TOF_td * SPD; // the TOF.Detla mostly lower than 30e-12 (usually I set it to 10e-12)
		float TOF_tr = tofinfo_flat(1); //TOF resolution for PET system (unit: s)
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

		float TOF_dif_time = event_flat(6);



		p1x = event_flat(0);// sou_p[0];
		p2x = event_flat(3);// end_p[0];
		pdx = p2x - p1x;

		p1y = event_flat(1);// sou_p[1];
		p2y = event_flat(4);// end_p[1];
		pdy = p2y - p1y;

		p1z = event_flat(2);// sou_p[2];
		p2z = event_flat(5);// end_p[2];
		pdz = p2z - p1z;

		Nx = grid_flat(0) + 1;
		Ny = grid_flat(1) + 1;
		Nz = grid_flat(2) + 1;

		alphax0 = (orgin_flat(0) - p1x) / (pdx + realmin);
		alphaxn = (orgin_flat(0) + (Nx - 1)*size_flat(0) - p1x) / (pdx + realmin);

		alphay0 = (orgin_flat(1) - p1y) / (pdy + realmin);
		alphayn = (orgin_flat(1) + (Ny - 1)*size_flat(1) - p1y) / (pdy + realmin);

		alphaz0 = (orgin_flat(2) - p1z) / (pdz + realmin);
		alphazn = (orgin_flat(2) + (Nz - 1)*size_flat(2) - p1z) / (pdz + realmin);



		alphaxmin = min(alphax0, alphaxn); alphaxmax = max(alphax0, alphaxn);
		alphaymin = min(alphay0, alphayn); alphaymax = max(alphay0, alphayn);
		alphazmin = min(alphaz0, alphazn); alphazmax = max(alphaz0, alphazn);

		alphatemp = max(alphaxmin, alphaymin);
		alphamin = max(alphatemp, alphazmin);

		alphatemp = min(alphaxmax, alphaymax);
		alphamax = min(alphatemp, alphazmax);

		if (alphamin < alphamax)
		{
			phyx_alphamin = (p1x + alphamin * pdx - orgin_flat(0)) / size_flat(0);
			phyx_alphamax = (p1x + alphamax * pdx - orgin_flat(0)) / size_flat(0);

			phyy_alphamin = (p1y + alphamin * pdy - orgin_flat(1)) / size_flat(1);
			phyy_alphamax = (p1y + alphamax * pdy - orgin_flat(1)) / size_flat(1);

			phyz_alphamin = (p1z + alphamin * pdz - orgin_flat(2)) / size_flat(2);
			phyz_alphamax = (p1z + alphamax * pdz - orgin_flat(2)) / size_flat(2);


			if (p1x < p2x)
			{
				if (alphamin == alphaxmin)    i_f = 1;
				else                          i_f = ceil(phyx_alphamin);
				if (alphamax == alphaxmax)    i_l = Nx - 1;
				else                          i_l = floor(phyx_alphamax);
				iu = 1;
				alphax = (orgin_flat(0) + i_f * size_flat(0) - p1x) / pdx;
			}
			else if (p1x > p2x)
			{
				if (alphamin == alphaxmin)    i_f = Nx - 2;
				else                          i_f = floor(phyx_alphamin);
				if (alphamax == alphaxmax)    i_l = 0;
				else                          i_l = ceil(phyx_alphamax);
				iu = -1;
				alphax = (orgin_flat(0) + i_f * size_flat(0) - p1x) / pdx;
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
				alphay = (orgin_flat(1) + j_f * size_flat(1) - p1y) / pdy;
			}
			else if (p1y > p2y)
			{
				if (alphamin == alphaymin)    j_f = Ny - 2;
				else                          j_f = floor(phyy_alphamin);
				if (alphamax == alphaymax)    j_l = 0;
				else                          j_l = ceil(phyy_alphamax);
				ju = -1;
				alphay = (orgin_flat(1) + j_f * size_flat(1) - p1y) / pdy;
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
				alphaz = (orgin_flat(2) + k_f * size_flat(2) - p1z) / pdz;
			}
			else if (p1z > p2z)
			{
				if (alphamin == alphazmin)    k_f = Nz - 2;
				else                          k_f = floor(phyz_alphamin);
				if (alphamax == alphazmax)    k_l = 0;
				else                          k_l = ceil(phyz_alphamax);
				ku = -1;
				alphaz = (orgin_flat(2) + k_f * size_flat(2) - p1z) / pdz;
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

			xindex = int(((p1x + alphaavg * pdx) - orgin_flat(0)) / size_flat(0));
			yindex = int(((p1y + alphaavg * pdy) - orgin_flat(1)) / size_flat(1));
			zindex = int(((p1z + alphaavg * pdz) - orgin_flat(2)) / size_flat(2));

			alphaxu = size_flat(0) / (fabs(pdx) + realmin);
			alphayu = size_flat(1) / (fabs(pdy) + realmin);
			alphazu = size_flat(2) / (fabs(pdz) + realmin);

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
						tmp_index.push_back((zindex * grid_flat(1) * grid_flat(0) + xindex * grid_flat(1) + yindex));
						tmp_data.push_back((alphax - alphac)*dconv*TOF);


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
						tmp_index.push_back((zindex * grid_flat(1) * grid_flat(0) + xindex * grid_flat(1) + yindex));
						tmp_data.push_back((alphay - alphac)*dconv*TOF);

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
						tmp_index.push_back((zindex * grid_flat(1) * grid_flat(0) + xindex * grid_flat(1) + yindex));
						tmp_data.push_back((alphaz - alphac)*dconv*TOF);
						zindex = zindex + ku;
						alphac = alphaz;
						alphaz = alphaz + alphazu;
					}
				}
			}

		}
	}


private:
	string model;

};

class Backprojection : public OpKernel
{
public:
	explicit Backprojection(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("model", &model));
	}

	void Compute(OpKernelContext *context) override
	{

		// Grab the geometries of an image.
		const Tensor &image = context->input(0);
		const Tensor &lors = context->input(1); // Grab the input lors.
		const Tensor &lor_values = context->input(2); //grab the input lor values
		const Tensor &tofinfo = context->input(3);
		const Tensor &grid = context->input(4);
		const Tensor &orgin = context->input(5);
		const Tensor &size = context->input(6);
		const Tensor &atan_map = context->input(7);


		// Create an output backprojected image
		Tensor *backpro_image = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(),
			&backpro_image));
		//set the initial backprojection image value to zero.
		auto backpro_image_flat = backpro_image->flat<float>();
		for (int i = 0; i < backpro_image_flat.size(); ++i) {
			backpro_image_flat(i) = 0;
		}



		

		backprojection(lors, lor_values, tofinfo, grid, orgin, size, backpro_image, atan_map);
	}

	void backprojection(const Tensor &events, const Tensor &lor_value,
		const Tensor &tofinfo, const Tensor &grid, const Tensor &orgin, const Tensor &size,
		Tensor* backpro_image, const Tensor &atan_map)
	{


		auto lor_value_flat = lor_value.flat<float>();

		auto atan_map_flat = atan_map.flat<float>();

		auto image_flat = backpro_image->flat<float>();
	

		std::vector<int> tmp_index;
		std::vector<float> tmp_data;
	
		
		int ind, ind_vol;
		//float atan_proj=0;
	

		for (ind = 0; ind < events.dim_size(0); ind++)
		{

			auto event = events.Slice(ind, ind + 1);
			float atan_proj = 0;

			CalculateFactor(events, tofinfo, grid, orgin, size, tmp_index,tmp_data);

			for (ind_vol = 0; ind_vol < tmp_index.size(); ind_vol++)
			{
				atan_proj += atan_map_flat(tmp_index[ind_vol]) * tmp_data[ind_vol];
				//std::cout<< tmp_data[ind_vol]<<endl;
			}

			for (ind_vol=0; ind_vol < tmp_index.size();ind_vol++)
			{
				image_flat(tmp_index[ind_vol]) += tmp_data[ind_vol] * lor_value_flat(ind)/exp(atan_proj);
				//std::cout<< tmp_data[ind_vol]<<endl;
			}
			
		}

		

	}

	void CalculateFactor(const Tensor &events, const Tensor &tofinfo, const Tensor &grid, const Tensor &orgin, const Tensor &size, std::vector<int> &tmp_index, std::vector<float> &tmp_data)
		/*input parametrer:
		coordinate in X,Y, Z axis of the start points of LOR: sou_p[3]
		coordinate in X,Y, Z axis of the start points of LOR: end_p[3]
		the struct to define the image volume( see the definition in header): VolPara
		the 3-D image volume: volume
		*/
	{
		auto tofinfo_flat = tofinfo.flat<float>();
		auto event_flat = events.flat<float>();
		auto grid_flat = grid.flat<int>();
		auto orgin_flat = orgin.flat<float>();
		auto size_flat = size.flat<float>();

	


		float DETLA_TOF = tofinfo_flat(0)*SPD;// TOF_td * SPD; // the TOF.Detla mostly lower than 30e-12 (usually I set it to 10e-12)
		float TOF_tr = tofinfo_flat(1); //TOF resolution for PET system (unit: s)
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

		float TOF_dif_time = event_flat(6);

		p1x = event_flat(0);// sou_p[0];
		p2x = event_flat(3);// end_p[0];
		pdx = p2x - p1x;

		p1y = event_flat(1);// sou_p[1];
		p2y = event_flat(4);// end_p[1];
		pdy = p2y - p1y;

		p1z = event_flat(2);// sou_p[2];
		p2z = event_flat(5);// end_p[2];
		pdz = p2z - p1z;

		Nx = grid_flat(0) + 1;
		Ny = grid_flat(1) + 1;
		Nz = grid_flat(2) + 1;

		alphax0 = (orgin_flat(0) - p1x) / (pdx + realmin);
		alphaxn = (orgin_flat(0) + (Nx - 1)*size_flat(0) - p1x) / (pdx + realmin);

		alphay0 = (orgin_flat(1) - p1y) / (pdy + realmin);
		alphayn = (orgin_flat(1) + (Ny - 1)*size_flat(1) - p1y) / (pdy + realmin);

		alphaz0 = (orgin_flat(2) - p1z) / (pdz + realmin);
		alphazn = (orgin_flat(2) + (Nz - 1)*size_flat(2) - p1z) / (pdz + realmin);

		alphaxmin = min(alphax0, alphaxn); alphaxmax = max(alphax0, alphaxn);
		alphaymin = min(alphay0, alphayn); alphaymax = max(alphay0, alphayn);
		alphazmin = min(alphaz0, alphazn); alphazmax = max(alphaz0, alphazn);

		alphatemp = max(alphaxmin, alphaymin);
		alphamin = max(alphatemp, alphazmin);

		alphatemp = min(alphaxmax, alphaymax);
		alphamax = min(alphatemp, alphazmax);

		if (alphamin < alphamax)
		{
			phyx_alphamin = (p1x + alphamin * pdx - orgin_flat(0)) / size_flat(0);
			phyx_alphamax = (p1x + alphamax * pdx - orgin_flat(0)) / size_flat(0);

			phyy_alphamin = (p1y + alphamin * pdy - orgin_flat(1)) / size_flat(1);
			phyy_alphamax = (p1y + alphamax * pdy - orgin_flat(1)) / size_flat(1);

			phyz_alphamin = (p1z + alphamin * pdz - orgin_flat(2)) / size_flat(2);
			phyz_alphamax = (p1z + alphamax * pdz - orgin_flat(2)) / size_flat(2);


			if (p1x < p2x)
			{
				if (alphamin == alphaxmin)    i_f = 1;
				else                          i_f = ceil(phyx_alphamin);
				if (alphamax == alphaxmax)    i_l = Nx - 1;
				else                          i_l = floor(phyx_alphamax);
				iu = 1;
				alphax = (orgin_flat(0) + i_f * size_flat(0) - p1x) / pdx;
			}
			else if (p1x > p2x)
			{
				if (alphamin == alphaxmin)    i_f = Nx - 2;
				else                          i_f = floor(phyx_alphamin);
				if (alphamax == alphaxmax)    i_l = 0;
				else                          i_l = ceil(phyx_alphamax);
				iu = -1;
				alphax = (orgin_flat(0) + i_f * size_flat(0) - p1x) / pdx;
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
				alphay = (orgin_flat(1) + j_f * size_flat(1) - p1y) / pdy;
			}
			else if (p1y > p2y)
			{
				if (alphamin == alphaymin)    j_f = Ny - 2;
				else                          j_f = floor(phyy_alphamin);
				if (alphamax == alphaymax)    j_l = 0;
				else                          j_l = ceil(phyy_alphamax);
				ju = -1;
				alphay = (orgin_flat(1) + j_f * size_flat(1) - p1y) / pdy;
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
				alphaz = (orgin_flat(2) + k_f * size_flat(2) - p1z) / pdz;
			}
			else if (p1z > p2z)
			{
				if (alphamin == alphazmin)    k_f = Nz - 2;
				else                          k_f = floor(phyz_alphamin);
				if (alphamax == alphazmax)    k_l = 0;
				else                          k_l = ceil(phyz_alphamax);
				ku = -1;
				alphaz = (orgin_flat(2) + k_f * size_flat(2) - p1z) / pdz;
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

			xindex = int(((p1x + alphaavg * pdx) - orgin_flat(0)) / size_flat(0));
			yindex = int(((p1y + alphaavg * pdy) - orgin_flat(1)) / size_flat(1));
			zindex = int(((p1z + alphaavg * pdz) - orgin_flat(2)) / size_flat(2));

			alphaxu = size_flat(0) / (fabs(pdx) + realmin);
			alphayu = size_flat(1) / (fabs(pdy) + realmin);
			alphazu = size_flat(2) / (fabs(pdz) + realmin);

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
						//volume[zindex * vol_num[1] * vol_num[0] + xindex * vol_num[1] + yindex] = (alphax - alphac)*dconv*TOF;
						//tmp_volume_flat(zindex * grid_flat(1) * grid_flat(0) + xindex * grid_flat(1) + yindex) = (alphax - alphac)*dconv*TOF;
						tmp_index.push_back((zindex * grid_flat(1) * grid_flat(0) + xindex * grid_flat(1) + yindex));
						tmp_data.push_back((alphax - alphac)*dconv*TOF);

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

						//volume[zindex * vol_num[1] * vol_num[0] + xindex * vol_num[1] + yindex] = (alphay - alphac)*dconv*TOF;
						//tmp_volume_flat(zindex * grid_flat(1) * grid_flat(0) + xindex * grid_flat(1) + yindex) = (alphay - alphac)*dconv*TOF;
						tmp_index.push_back((zindex * grid_flat(1) * grid_flat(0) + xindex * grid_flat(1) + yindex));
						tmp_data.push_back((alphay - alphac)*dconv*TOF);
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

						//volume[zindex * vol_num[1] * vol_num[0] + xindex * vol_num[1] + yindex] = (alphaz - alphac)*dconv*TOF;
						//tmp_volume_flat(zindex * grid_flat(1) * grid_flat(0) + xindex * grid_flat(1) + yindex) = (alphaz - alphac)*dconv*TOF;
						tmp_index.push_back((zindex * grid_flat(1) * grid_flat(0) + xindex * grid_flat(1) + yindex));
						tmp_data.push_back((alphaz - alphac)*dconv*TOF);
						zindex = zindex + ku;
						alphac = alphaz;
						alphaz = alphaz + alphazu;
					}
				}
			}

		}
	}

	

private:
	string model;
};

#define REGISTER_CPU_KERNEL(name, op) \
  REGISTER_KERNEL_BUILDER(            \
      Name(name).Device(DEVICE_CPU), op)

REGISTER_CPU_KERNEL("Projection", Projection);
REGISTER_CPU_KERNEL("Backprojection", Backprojection);

#undef REGISTER_CPU_KERNEL