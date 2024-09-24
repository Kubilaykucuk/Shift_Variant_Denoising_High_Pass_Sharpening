#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

cv::Mat fastBilateralFilter(const cv::Mat& src, int d, double sigmaColor, double sigmaSpace) {
    cv::Mat src2;
    src.convertTo(src2, CV_32F);
    cv::Mat dst;
    cv::bilateralFilter(src2, dst, d, sigmaColor, sigmaSpace, cv::BORDER_REFLECT_101);
    return dst;
}

cv::Mat conv2D(const cv::Mat& image, const cv::Mat& kernel) 
{
    int kRows = kernel.rows;
    int kCols = kernel.cols;
    
    int padX = kCols / 2;
    int padY = kRows / 2;

    cv::Mat image32F;
    if (image.type() != CV_32F) {
        image.convertTo(image32F, CV_32F);
    } else {
        image32F = image;
    }

    cv::Mat paddedImage;
    cv::copyMakeBorder(image32F, paddedImage, padY, padY, padX, padX, cv::BORDER_REPLICATE);

    cv::Mat output;
    int channels = image.channels();

    if(channels == 1) 
    {
        output.create(image.size(), CV_32F);
    } else 
    {
        output.create(image.size(), CV_32FC3);
    }

    #pragma omp parallel for
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            if (channels == 1) {
                float sum = 0.0f;
                for (int m = 0; m < kRows; ++m) {
                    for (int n = 0; n < kCols; ++n) {
                        sum += kernel.at<float>(m, n) * paddedImage.at<float>(i + m, j + n);
                    }
                }
                output.at<float>(i, j) = sum;
            } else {
                cv::Vec3f sum(0.0f, 0.0f, 0.0f);

                for (int m = 0; m < kRows; ++m) {
                    for (int n = 0; n < kCols; ++n) {
                        cv::Vec3f pixel = paddedImage.at<cv::Vec3f>(i + m, j + n);
                        float weight = kernel.at<float>(m, n);
                        
                        sum[0] += weight * pixel[0];
                        sum[1] += weight * pixel[1];
                        sum[2] += weight * pixel[2];
                    }
                }

                output.at<cv::Vec3f>(i, j) = sum;
            }
        }
    }

    return output;
}

cv::Mat createGaussianKernel(int kernelSize, double sigma) {
    // 1D Gaussian Kernel
    cv::Mat kernel1D(kernelSize, 1, CV_32F);
    int center = kernelSize / 2;
    double sum = 0.0;

    for (int i = 0; i < kernelSize; ++i) {
        double x = i - center;
        kernel1D.at<float>(i, 0) = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel1D.at<float>(i, 0);
    }

    // Normalize the 1D kernel
    for (int i = 0; i < kernelSize; ++i) {
        kernel1D.at<float>(i, 0) /= sum;
    }

    // Create the 2D Gaussian Kernel
    cv::Mat kernel2D(kernelSize, kernelSize, CV_32F);
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel2D.at<float>(i, j) = kernel1D.at<float>(i, 0) * kernel1D.at<float>(j, 0);
        }
    }

    return kernel2D;
}

cv::Mat manualBilateralFilter(const cv::Mat& src, int d, double sigmaColor, double sigmaSpace) {
    int radius = d / 2;
    
    cv::Mat dst = src.clone();
    

    // Precompute spatial weights
    std::vector<double> spatialWeights((d + 1) * (d + 1));
    double invSigmaSpace2 = 1.0 / (2.0 * sigmaSpace * sigmaSpace);
    double invSigmaColor2 = 1.0 / (2.0 * sigmaColor * sigmaColor);
    
    #pragma omp parallel for
    for (int k = -radius; k <= radius; ++k) {
        for (int l = -radius; l <= radius; ++l) {
            double dist = k * k + l * l;
            spatialWeights[(k + radius) * d + (l + radius)] = std::exp(-dist * invSigmaSpace2);
        }
    }
    


    #pragma omp parallel for
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            double sumWeights = 0.0;
            cv::Vec3f sumColors(0, 0, 0);
            cv::Vec3f pixelValue = src.at<cv::Vec3f>(i, j);

            // Apply bilateral filter
            for (int k = -radius; k <= radius; ++k) {
                for (int l = -radius; l <= radius; ++l) {
                    int y = std::min(std::max(i + k, 0), src.rows - 1);
                    int x = std::min(std::max(j + l, 0), src.cols - 1);

                    cv::Vec3f neighborValue = src.at<cv::Vec3f>(y, x);
                    double spatialWeight = spatialWeights[(k + radius) * d + (l + radius)];
                    cv::Vec3f diff = pixelValue - neighborValue;
                    double colorWeight = std::exp(-(diff.dot(diff)) * invSigmaColor2);
                    double weight = spatialWeight * colorWeight;

                    sumWeights += weight;
                    sumColors += weight * neighborValue;
                }
            }

            dst.at<cv::Vec3f>(i, j) = sumColors / sumWeights;
        }
    }

    return dst;
}

cv::Mat highPassFilter(const cv::Mat& src, int kernelSize) {
    cv::Mat srcFloat;
    if (src.type() != CV_32F) {
        src.convertTo(srcFloat, CV_32F);
    } else {
        srcFloat = src;
    }
    
    cv::Mat gaussianKernel = createGaussianKernel(kernelSize, (kernelSize - 1) / 2);

    cv::Mat lowPass = conv2D(srcFloat, gaussianKernel);

    cv::Mat highPass;
    cv::subtract(srcFloat, lowPass, highPass);

    

    cv::normalize(highPass, highPass, 0, 255, cv::NORM_MINMAX);
    highPass.convertTo(highPass, CV_8U);

    return highPass;
}

cv::Mat totalVariationDenoising(const cv::Mat& src, float lambda, int iterations, float tau) {
    int rows = src.rows;
    int cols = src.cols;
    int channels = src.channels();

    cv::Mat u = src.clone();
    cv::Mat px = cv::Mat::zeros(src.size(), src.type());
    cv::Mat py = cv::Mat::zeros(src.size(), src.type());

    cv::Mat uShiftRight, uShiftDown, pxShiftLeft, pyShiftUp;
    cv::copyMakeBorder(u, uShiftRight, 0, 0, 1, 0, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(u, uShiftDown, 1, 0, 0, 0, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(px, pxShiftLeft, 0, 0, 0, 1, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(py, pyShiftUp, 0, 1, 0, 0, cv::BORDER_REPLICATE);

    float factor = tau / lambda;

    #pragma omp parallel for
    for (int i = 0; i < iterations; ++i) {
        // Compute gradients
        cv::Mat ux, uy;

        ux = uShiftRight(cv::Rect(1, 0, cols, rows)) - u;
        uy = uShiftDown(cv::Rect(0, 1, cols, rows)) - u;

        // Update dual variables
        px = px + factor * ux;
        py = py + factor * uy;

        // Compute projection
        cv::Mat norm;
        cv::sqrt(px.mul(px) + py.mul(py), norm);
        norm = cv::max(1.0f, norm);
        px = px / norm;
        py = py / norm;

        // Divergence
        cv::Mat div;

        div = (px - pxShiftLeft(cv::Rect(1, 0, cols, rows))) + (py - pyShiftUp(cv::Rect(0, 1, cols, rows)));

        // Update primal variable
        u = src - lambda * div;
    }
    return u;
}

double computeSSIM(const cv::Mat& I1, const cv::Mat& I2)
{
    const double C1 = 6.5025, C2 = 58.5225;

    // Convert input images to CV_32F
    cv::Mat I1_f, I2_f;
    I1.convertTo(I1_f, CV_32F);
    I2.convertTo(I2_f, CV_32F);

    cv::Mat I1_2 = I1_f.mul(I1_f);
    cv::Mat I2_2 = I2_f.mul(I2_f);
    cv::Mat I1_I2 = I1_f.mul(I2_f);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1_f, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2_f, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);

    cv::Scalar mssim = cv::mean(ssim_map);

    return mssim[0];
}

cv::Mat processImage(const cv::Mat& src) {
    int d = 9;
    double sigmaColor = 75.0;
    double sigmaSpace = 75.0;

    cv::Mat denoised;
    cv::Mat srcF;
    src.convertTo(srcF,CV_32F);
    float lambda = 0.125f;
    int iterations = 5;
    float tau = 0.25f;
    if (srcF.channels() == 1) {
        denoised = totalVariationDenoising(srcF, lambda, iterations, tau);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(srcF, channels);
        for (auto& channel : channels) {
            channel = totalVariationDenoising(channel, lambda, iterations, tau);
        }
        cv::merge(channels, denoised);
    }

    // cv::Mat denoised = manualBilateralFilter(src, d, sigmaColor, sigmaSpace);
    // cv::Mat denoised = fastBilateralFilter(src, d, sigmaColor, sigmaSpace);

    int kernelSize = 5;
    cv::Mat highPass = highPassFilter(denoised, kernelSize);

    cv::Mat denoisedFloat, highPassFloat;
    denoised.convertTo(denoisedFloat, CV_32F);
    highPass.convertTo(highPassFloat, CV_32F);

    cv::Mat result;
    double alpha = 1.0;
    double beta = 0.6;
    cv::addWeighted(denoisedFloat, alpha, highPassFloat, beta, 0, result);


    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8U);

    return result;
}

int main() {
    omp_set_num_threads(20);
    std::string videoPath = "test_video.mp4";
    cv::VideoCapture cap(videoPath);
    cv::Mat img, grayImg;

    const int frameCountToAverage = 100;
    int frameCount = 0;
    double totalProcessingTime = 0.0;
    
    while (true) {
        cap >> img;
        if (img.empty()) {
            break;
        }

        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat processedImg = processImage(grayImg);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> actual_time = end - start;
        double processingTime;
        if(actual_time.count() == 0)
        {
            processingTime = 0.0005;
        }
        else
        {
            processingTime = actual_time.count();
        }
        
        totalProcessingTime += processingTime;
        frameCount++;
        double averageFPS;
        if (frameCount >= frameCountToAverage) {
            averageFPS = frameCountToAverage / totalProcessingTime;
            std::cout << "Average FPS over " << frameCountToAverage << " frames: " << averageFPS << std::endl;
            frameCount = 0;
            totalProcessingTime = 0.0;
        }

        cv::Mat processedImgFloat, imgFloat;
        processedImg.convertTo(processedImgFloat, CV_32F);
        grayImg.convertTo(imgFloat, CV_32F);
        double ssim_value = computeSSIM(processedImgFloat, imgFloat);

        std::ostringstream ssimText;
        ssimText << "SSIM Score: " << ssim_value;

        std::ostringstream fpsText;
        fpsText << "Average FPS: " << averageFPS;

        cv::Mat displayImage;
        processedImg.copyTo(displayImage);
        cv::putText(displayImage, ssimText.str(), cv::Point(10, displayImage.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(displayImage,fpsText.str(),cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Original Image", img);
        cv::imshow("Processed Image", displayImage);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
}