
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/*
 * The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size.
 * However, you can make this function work for other sizes too.
 * For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
 */
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Temporary storage for matches within the bounding box
    std::vector<cv::DMatch> matchesInROI;

    for (auto match : kptMatches)
    {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            matchesInROI.push_back(match);
        }
    }

    // Calculate distances
    std::vector<double> distances;

    for (auto match : matchesInROI)
    {
        cv::Point2f ptPrev = kptsPrev[match.queryIdx].pt;
        cv::Point2f ptCurr = kptsCurr[match.trainIdx].pt;
        double dist = cv::norm(ptCurr - ptPrev);
        distances.push_back(dist);
    }

    // Calculate the median distance
    std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
    double medianDist = distances[distances.size() / 2];

    // Calculate the median absolute deviation (MAD)
    std::vector<double> absDevs;
    for (double dist : distances)
    {
        absDevs.push_back(std::abs(dist - medianDist));
    }

    std::nth_element(absDevs.begin(), absDevs.begin() + absDevs.size() / 2, absDevs.end());
    double mad = absDevs[absDevs.size() / 2];

    // Define a threshold to filter out outliers based on MAD
    double threshold = medianDist + 3 * mad;

    // Filter matches to remove outliers
    boundingBox.kptMatches.clear();

    for (size_t i = 0; i < matchesInROI.size(); ++i)
    {
        if (distances[i] <= threshold)
        {
            boundingBox.kptMatches.push_back(matchesInROI[i]);
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    } // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // EOF STUDENT TASK
}

double getPercentile(std::vector<double> &data, double percentile)
{
    if (data.empty())
        return 0.0;

    size_t n = data.size();
    size_t index = static_cast<size_t>(percentile * n);
    std::nth_element(data.begin(), data.begin() + index, data.end());
    return data[index];
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1.0 / frameRate; // time between two measurements in seconds

    // Extract x-coordinates from Lidar points
    std::vector<double> xPrev, xCurr;
    for (const auto &point : lidarPointsPrev)
    {
        xPrev.push_back(point.x);
    }
    for (const auto &point : lidarPointsCurr)
    {
        xCurr.push_back(point.x);
    }

    // Filter out outliers using percentiles
    double lowerPercentile = 0.1; // 10th percentile
    double upperPercentile = 0.9; // 90th percentile

    double xPrevLower = getPercentile(xPrev, lowerPercentile);
    double xPrevUpper = getPercentile(xPrev, upperPercentile);
    double xCurrLower = getPercentile(xCurr, lowerPercentile);
    double xCurrUpper = getPercentile(xCurr, upperPercentile);

    // Filter x-coordinates to keep only values within the 10th to 90th percentile range
    auto filterPercentiles = [](std::vector<double> &data, double lower, double upper)
    {
        data.erase(std::remove_if(data.begin(), data.end(),
                                  [lower, upper](double x)
                                  { return x < lower || x > upper; }),
                   data.end());
    };

    filterPercentiles(xPrev, xPrevLower, xPrevUpper);
    filterPercentiles(xCurr, xCurrLower, xCurrUpper);

    // Compute the minimum x-coordinate after filtering
    double minXPrev = *std::min_element(xPrev.begin(), xPrev.end());
    double minXCurr = *std::min_element(xCurr.begin(), xCurr.end());

    // Compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
    std::cout << "minXCurr: " << minXCurr << endl;
    std::cout << "minXPrev: " << minXPrev << endl;
    std::cout << "minXPrev - minXCurr: " << minXPrev - minXCurr << endl;
    // // Compute the median x-coordinate after filtering
    // std::sort(xPrev.begin(), xPrev.end());
    // std::sort(xCurr.begin(), xCurr.end());

    // double medianXPrev = xPrev[xPrev.size() / 2];
    // double medianXCurr = xCurr[xCurr.size() / 2];

    // // Compute TTC from both measurements
    // TTC = medianXCurr * dT / (medianXPrev - medianXCurr);
}

void getMaxCountPairs(const std::map<std::pair<int, int>, int> &bbMatchCounts, std::map<int, int> &bbBestMatches)
{
    // Map to track the maximum count for each firstBoxId
    std::map<int, std::pair<int, int>> maxCounts;

    // Iterate through bbMatchCounts to find the maximum counts per firstBoxId
    for (const auto &bbMatch : bbMatchCounts)
    {
        int firstBoxId = bbMatch.first.first;
        int secondBoxId = bbMatch.first.second;
        int count = bbMatch.second;

        if (maxCounts.find(firstBoxId) == maxCounts.end() || count > maxCounts[firstBoxId].second)
        {
            maxCounts[firstBoxId] = std::make_pair(secondBoxId, count);
        }
    }

    // Create the final map with the best matches
    // std::map<int, int> bbBestMatches;
    for (const auto &maxCount : maxCounts)
    {
        int firstBoxId = maxCount.first;
        int secondBoxId = maxCount.second.first;
        bbBestMatches[firstBoxId] = secondBoxId;
        std::cout << "Best match for " << firstBoxId << ": (" << firstBoxId << ", " << secondBoxId << ") with count " << maxCount.second.second << std::endl;
    }

    // return bbBestMatches;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{

    std::map<std::pair<int, int>, int> bbMatchCounts;

    // Loop over all matches
    for (auto &match : matches)
    {
        cv::KeyPoint prevKeypoint = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKeypoint = currFrame.keypoints[match.trainIdx];

        // Loop over all bounding boxes in the previous frame
        for (auto &prevBB : prevFrame.boundingBoxes)
        {
            // Loop over all bounding boxes in the current frame
            for (auto &currBB : currFrame.boundingBoxes)
            {
                // Check if keypoints fall within the bounding boxes
                if (prevBB.roi.contains(prevKeypoint.pt) && currBB.roi.contains(currKeypoint.pt))
                {
                    std::pair<int, int> bbPair = std::make_pair(prevBB.boxID, currBB.boxID);
                    // Output current state before incrementing
                    // std::cout << "Before incrementing: " << bbMatchCounts[bbPair] << std::endl;

                    // Increment the count for bbPair
                    bbMatchCounts[bbPair]++;

                    // Output the count after incrementing
                    // std::cout << "After incrementing: " << bbMatchCounts[bbPair] << std::endl;
                    // std::cout << bbMatchCounts[bbPair]
                    // std::cout << "PrevBB: " << bbMatchCounts.first.first << ", CurrBB: " << bbMatchCounts.first.second << " => Count: " << bbMatchCounts.second << std::endl;
                }
            }
        }
    }
    getMaxCountPairs(bbMatchCounts, bbBestMatches);
}
