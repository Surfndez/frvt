#include <algorithm>

#include <torch/script.h>

#include "FanLandmarksDetector.h"
#include "Inverse.h"

using namespace FRVT_11;

int REFERENCE_SCALE = 195;

at::Tensor InverseMatrix(float** mat)
{
    int mat[3][3], i, j;
	
    // Enter elements of matrix row wise
	for(i = 0; i < 3; i++) {
		for(j = 0; j < 3; j++) {
            mat[i][j] = matTensor[i][j].item<float>();
        }
    }

    //finding determinant
    float determinant = 0;
	for(i = 0; i < 3; i++)
		determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));

    // Inverse matrix
    std::vector<float> inverseMat;
	for(i = 0; i < 3; i++) {
		for(j = 0; j < 3; j++) {
			auto v = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3]))/ determinant;
            inverseMat.push_back(v);
        }
    }

    auto newMatTensor = torch::tensor(inverseMat, torch::requires_grad(false).dtype(torch::kFloat32)).view({3, 3});

    return newMatTensor;
}

at::Tensor
transform(std::vector<float> point, at::Tensor center, int scale, float resolution, bool invert=false)
{
    std::vector<float> _pt = {1, 1, 1};
    _pt[0] = point[0];
    _pt[1] = point[1];

    auto h = 200.0 * scale;
    std::vector<float> t = {1, 0, 0, /**/ 0, 1, 0, /**/ 0, 0, 1};
    t[0][0] = resolution / h;
    t[1][1] = resolution / h;
    t[0][2] = resolution * (-center[0] / h + 0.5);
    t[1][2] = resolution * (-center[1] / h + 0.5);

    if (invert)
        t = InverseMatrix(t);

    auto new_point = at::matmul(t, _pt);
    new_point = new_point.slice(0, 0, 2); //[0:2];

    return new_point;
}

at::Tensor
CropImage(at::Tensor &tensorImage, int* center, int scale , float resolution=256.0f)
{
    std::cout << "CropImage 0" << std::endl;

    // Crop around the center point
    /* Crops the image around the center. Input is expected to be an np.ndarray */
    auto ul = transform({1, 1}, center, scale, resolution, true);
    auto br = transform({resolution, resolution}, center, scale, resolution, true);

    std::cout << "CropImage 1" << std::endl;

    auto newImg = at::zeros({1, int(*(br[1] - ul[1]).data<float>()), int(*(br[0] - ul[0]).data<float>()), tensorImage.sizes()[3]}, at::ScalarType::Byte);

    auto ht = tensorImage.sizes()[1];
    auto wd = tensorImage.sizes()[2];

    std::cout << "CropImage 2" << std::endl;

    auto newX = {std::max(1, -int(*(ul[0].data<float>())) + 1), std::min(int(*br[0].data<float>()), int(wd)) - int(*ul[0].data<float>())};
    auto newY = {std::max(1, -int(*(ul[1].data<float>())) + 1), std::min(int(*br[1].data<float>()), int(ht)) - int(*ul[1].data<float>())};

    std::cout << "CropImage 3" << std::endl;

    //auto oldX = {}; // np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    //auto oldY = {}; // np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)

    return tensorImage;
}

FanLandmarksDetector::FanLandmarksDetector(const std::string &configDir)
{
    std::string landmarksDetectorModelPath = configDir + "/fan.pt";

    // Deserialize the ScriptModule from a file using torch::jit::load().
    mLandmarksDetector = torch::jit::load(landmarksDetectorModelPath);
    assert(mLandmarksDetector != nullptr);
}

FanLandmarksDetector::~FanLandmarksDetector() {}

std::vector<int>
FanLandmarksDetector::Detect(const ImageData& imageData, const Rect &face) const
{
    std::cout << "Detect landmarks... ";

    // create image tensor
    std::vector<int64_t> sizes = {1, imageData.height, imageData.width, imageData.channels};
    at::TensorOptions options(at::ScalarType::Byte);
    at::Tensor tensorImage = torch::from_blob(imageData.data.get(), at::IntList(sizes), options);
    tensorImage = tensorImage.toType(at::kFloat);

    std::cout << " 3 ";

    // calculate center and scale
    int d[] = {face.x1, face.y1, face.x2, face.y2};
    float center[] = {float(d[2] - (d[2] - d[0]) / 2.0), float(d[3] - (d[3] - d[1]) / 2.0)};
    center[1] = center[1] - (d[3] - d[1]) * 0.12;
    float scale = (d[2] - d[0] + d[3] - d[1]) / REFERENCE_SCALE;

    std::cout << " 2 ";

    // Crop image
    tensorImage = CropImage(tensorImage, center, scale);

    // HWC -> CHW
    tensorImage = tensorImage.permute({0, 3, 1, 2});

    // Normalize image
    tensorImage = tensorImage.div(255.0).unsqueeze_(0);

    std::cout << " 1 ";

    // Inference
    torch::jit::IValue output = mLandmarksDetector->forward({tensorImage});

    // Adjust output
    //pts, pts_img = get_preds_fromhm(out, center, scale)
    //pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
    //infer_detection(detections[0])

    std::cout << "Done!" << std::endl;

    std::vector<int> landmarks;
    return landmarks;
}
