#ifndef OTF_H
#define OTF_H

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "Camera.h"
#include "Matrix.h"
#include "STBCommons.h"
#include "myMATH.h"

class TracerConfig;

struct OTFParam
{
    double dx, dy, dz;
    int n_cam, nx, ny, nz, n_grid; // n_grid = nx*ny*nz

    // OTF parameters: calculate intensity of a tracer onto camera
    //  2D projection point: (x0, y0)
    //  xx =  (x-x0) * cos(alpha) + (y-y0) * sin(alpha)
    //  yy = -(x-x0) * sin(alpha) + (y-y0) * cos(alpha)
    //  img[y,x] = a * exp(-b * xx^2 - c * yy^2) 
    // default: a = 125, b = 1.5, c = 1.5, alpha = 0   
    Matrix<double> a, b, c, alpha;

    // The image model (the cylinder STB plan, Phase 2):
    //  0, "gauss": the Gaussian above;
    //  1, "segment": a rectangle of length |b| along xx and |c| along yy [px]
    //     (signed lengths, so that the table interpolates through focus),
    //     blurred by an isotropic Gaussian of standard deviation sigma [px]:
    //     img = a * g(xx; b) * g(yy; c), g(t; L) = [Phi((t + L/2)/sigma) -
    //     Phi((t - L/2)/sigma)] / [2 Phi(L/(2 sigma)) - 1], which is
    //     exp(-t^2/(2 sigma^2)) as L -> 0; a is the peak. Lines (b or c about
    //     0) and round spots are both its cases.
    int model = 0;
    Matrix<double> sigma; // model 1 only 
    
    AxisLimit boundary;
    std::vector<double> grid_x, grid_y, grid_z;
};

class OTF 
{
private:
    //  i in 0 ~ n_grid: i = x_id * (ny*nz) + y_id * nz + z_id
    //  for x_id = 0:_nx-1 
    //      for y_id = 0:_ny-1 
    //          for z_id = 0:_nz-1  
    //              i ++
    int mapGridID (int x_id, int y_id, int z_id) const
    {
        int id = x_id * (_param.ny*_param.nz) + y_id * _param.nz + z_id;
        return id;
    }
    void setGrid ();

public:
    OTFParam _param;
    std::string _output_path; // output path for OTF parameter

    OTF () {};
    OTF(const OTF& otf) : _param(otf._param) {};

    // nx,ny,nz >= 2
    // set as default a,b,c,alpha values
    OTF(int n_cam, int nx, int ny, int nz, AxisLimit const& boundary,
        const std::vector<std::shared_ptr<Camera>>& camera_models); 

    // nx,ny,nz >= 2
    // use file to load a,b,c,alpha
    OTF(std::string otf_file); 

    // get a rough OTF using images, the parameters are uniform across the 3D space, but just adjusted to the 2D real image
    void estimateUniformOTFFromImage(int cam_id,
                                 TracerConfig& tracer_config,
                                 const std::vector<Image>& img_list);

    ~OTF() {};

    void loadParam (int n_cam, int nx, int ny, int nz, AxisLimit const& boundary,
                    const std::vector<std::shared_ptr<Camera>>& camera_models);
    void loadParam (std::string otf_file);

    void saveParam (std::string otf_file);

    // Output: (a, b, c, cos(alpha), sin(alpha)), and sigma for the segment model
    std::vector<double> getOTFParam(int cam_id, Pt3D const& pt_world) const;

    // Half-width and half-height [px] of the box around the image centre that
    // holds the image out to k standard deviations of its blur (Gaussian: the
    // k-sigma ellipse; segment: the rectangle grown by k sigma), rotated by
    // alpha, at pt_world. params: getOTFParam()'s output.
    static void halfExtent(int model, const std::vector<double>& params, double k,
                           double& dx, double& dy);
    void halfExtent(int cam_id, Pt3D const& pt_world, double k, double& dx,
                    double& dy) const;

    // The image intensity at (x, y) of an image centred at (x0, y0) with
    // getOTFParam()'s params, for either model.
    static double intensity(int model, double x, double y, double x0, double y0,
                            const std::vector<double>& params);

};

#endif
