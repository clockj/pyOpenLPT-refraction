#ifndef OBJECTFINDER_H
#define OBJECTFINDER_H

#include <vector>
#include <typeinfo>

#include "Config.h"
#include "ObjectInfo.h"
#include "Matrix.h"
#include "STBCommons.h"
#include "myMATH.h"

#include "CircleIdentifier.h"

class ObjectFinder2D
{
public:
    ObjectFinder2D() = default;
    ~ObjectFinder2D() = default;

    // Find 2D objects in the image based on the object configuration; cam_id
    // (when known) picks per-camera settings (the line detector's noise)
    std::vector<std::unique_ptr<Object2D>>
    findObject2D(Image const& img, ObjectConfig const& obj_cfg, int cam_id = -1);

private:
    std::vector<std::unique_ptr<Object2D>>
    findTracer2D(Image const& img, TracerConfig const& cfg, int cam_id = -1);

    // tracer.detector "lines": LineFinder::detect (LineFinder.h), each object
    // a Tracer2D with its shape (_has_shape, _len_h_px, _len_v_px, ...)
    std::vector<std::unique_ptr<Object2D>>
    findTracerLines2D(Image const& img, TracerConfig const& cfg, int cam_id);

    std::vector<std::unique_ptr<Object2D>>
    findBubble2D(Image const& img, BubbleConfig const& cfg);
};


#endif