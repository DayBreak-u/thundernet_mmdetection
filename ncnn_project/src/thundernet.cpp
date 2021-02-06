// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include<iostream>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};


static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// copy from src/layer/proposal.cpp
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++)
    {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++)
        {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

static void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, std::vector<Object>& objects)
{
    int w = score_blob.w;
    int h = score_blob.h;

    // generate  proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q = 0; q < num_anchors; q++)
    {
        const float* anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++)
        {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++)
            {
                int index = i * w + j;

                float prob = score[index];



                // apply center size
                float dx = bbox.channel(0)[index];
                float dy = bbox.channel(1)[index];
                float dw = bbox.channel(2)[index];
                float dh = bbox.channel(3)[index];

                float cx = anchor_x + anchor_w * 0.5f;
                float cy = anchor_y + anchor_h * 0.5f;

                float pb_cx = cx + anchor_w * dx;
                float pb_cy = cy + anchor_h * dy;

                float pb_w = anchor_w * exp(dw);
                float pb_h = anchor_h * exp(dh);

                float x0 = pb_cx - pb_w * 0.5f;
                float y0 = pb_cy - pb_h * 0.5f;
                float x1 = pb_cx + pb_w * 0.5f;
                float y1 = pb_cy + pb_h * 0.5f;

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0 + 1;
                obj.rect.height = y1 - y0 + 1;

                obj.prob = prob;

                objects.push_back(obj);


                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
}

static inline float bilinear_interpolate(const float* ptr, int w, int h, float x, float y)
{
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    float a0 = x1 - x;
    float a1 = x - x0;
    float b0 = y1 - y;
    float b1 = y - y0;

    if (x1 >= w)
    {
        x1 = w - 1;
        a0 = 1.f;
        a1 = 0.f;
    }
    if (y1 >= h)
    {
        y1 = h - 1;
        b0 = 1.f;
        b1 = 0.f;
    }

    float r0 = ptr[y0 * w + x0] * a0 + ptr[y0 * w + x1] * a1;
    float r1 = ptr[y1 * w + x0] * a0 + ptr[y1 * w + x1] * a1;

    float v = r0 * b0 + r1 * b1;

    return v;
}


int psroialign( ncnn::Mat bottom_blob,ncnn::Mat  roi_blob, ncnn::Mat& top_blob)
{

    float spatial_scale = 1./16;
    int pooled_height = 7;
    int pooled_width = 7;
    int output_dim = 5;
    int sampling_ratio = 2;


    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int channels = bottom_blob.c;

    if (channels != output_dim * pooled_width * pooled_height)
    {
        // input channel number does not match layer parameters
        return -1;
    }

    top_blob.create(pooled_width, pooled_height, output_dim,elemsize);
    if (top_blob.empty())
        return -100;

    // For each ROI R = [x y w h]: avg pool over R
    const float* roi_ptr = roi_blob;

    float roi_x1 = static_cast<float>(round(roi_ptr[0]) * spatial_scale);
    float roi_y1 = static_cast<float>(round(roi_ptr[1]) * spatial_scale);
    float roi_x2 = static_cast<float>(round(roi_ptr[2] + 1.f) * spatial_scale);
    float roi_y2 = static_cast<float>(round(roi_ptr[3] + 1.f) * spatial_scale);

    float roi_w = std::max(roi_x2 - roi_x1, 0.1f);
    float roi_h = std::max(roi_y2 - roi_y1, 0.1f);

    float bin_size_w = roi_w / (float)pooled_width;
    float bin_size_h = roi_h / (float)pooled_height;

    #pragma omp parallel for num_threads(8)
    for (int q = 0; q < output_dim; q++)
    {
        float* outptr = top_blob.channel(q);

        for (int ph = 0; ph < pooled_height; ph++)
        {
            for (int pw = 0; pw < pooled_width; pw++)
            {
                const float* ptr = bottom_blob.channel((q * pooled_height + ph) * pooled_width + pw);

                int hstart = static_cast<int>(floor(roi_y1 + (float)(ph)*bin_size_h));
                int wstart = static_cast<int>(floor(roi_x1 + (float)(pw)*bin_size_w));
                int hend = static_cast<int>(ceil(roi_y1 + (float)(ph + 1) * bin_size_h));
                int wend = static_cast<int>(ceil(roi_x1 + (float)(pw + 1) * bin_size_w));



                hstart = std::min(std::max(hstart, 0), h);
                wstart = std::min(std::max(wstart, 0), w);
                hend = std::min(std::max(hend, 0), h);
                wend = std::min(std::max(wend, 0), w);


                int bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(hend - hstart));
                int bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(wend - wstart));


                bool is_empty = (hend <= hstart) || (wend <= wstart);
                int area = (hend - hstart) * (wend - wstart);

                float sum = 0.f;

                for (int by = 0; by < bin_grid_h; by++)
                {
                    float y = hstart + (by + 0.5f) * bin_size_h / (float)bin_grid_h;

                    for (int bx = 0; bx < bin_grid_w; bx++)
                    {
                        float x = wstart + (bx + 0.5f) * bin_size_w / (float)bin_grid_w;

                        // bilinear interpolate at (x,y)
                        float v = bilinear_interpolate(ptr, w, h, x, y);

                        sum += v;
                    }
                }
//                for (int y = hstart; y < hend; y++)
//                {
//                    for (int x = wstart; x < wend; x++)
//                    {
//                        int index = y * w + x;
//                        sum += ptr[index];
//                    }
//                }

                outptr[pw] = is_empty ? 0.f : (sum / (float)area);
            }

            outptr += pooled_width;
        }
    }

    return 0;
}




static int detect_thundernet(const cv::Mat& bgr,std::vector<Object>& objects)
{
    ncnn::Net thundernet;
    ncnn::Net thundernet_rcnn;

    thundernet.opt.use_vulkan_compute = true;
    thundernet_rcnn.opt.use_vulkan_compute = true;


    thundernet.load_param("../models/thundernet_mbv2_rpn-opt-fp16.param");
    thundernet.load_model("../models/thundernet_mbv2_rpn-opt-fp16.bin");

    thundernet_rcnn.load_param("../models/thundernet_mbv2_rcnn-opt-fp16.param");
    thundernet_rcnn.load_model("../models/thundernet_mbv2_rcnn-opt-fp16.bin");

    const int pre_nms_topN = 3000;
    const int after_nms_topN = 200;
    const float nms_rpn = 0.7f;
    const float nms_threshold = 0.5f;
    const float confidence_thresh = 0.4f;
    const int max_per_image = 100;
    const int target_size = 320;


    int img_w = bgr.cols;
    int img_h = bgr.rows;
    float scale_w = (float)target_size / img_w;
    float scale_h = (float)target_size / img_h;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,img_w,img_h ,target_size, target_size);

    const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
    const float norm_vals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = thundernet.create_extractor();

    ex.input("input", in);

    std::vector<Object> proposal_boxes;


    ncnn::Mat score_blob, bbox_blob, feat;
    ex.extract("rpn_cls_score", score_blob);
    ex.extract("rpn_bbox_pred", bbox_blob);
    ex.extract("x", feat);


    const int base_size = 16;
    const int feat_stride = 16;
    ncnn::Mat ratios(5);
    ratios[0] = 0.5f;
    ratios[1] = 0.75f;
    ratios[2] = 1.f;
    ratios[3] = 1.333f;
    ratios[4] = 2.f;
    ncnn::Mat scales(5);
    scales[0] = 2.f;
    scales[1] = 4.f;
    scales[2] = 8.f;
    scales[3] = 16.f;
    scales[4] = 32.f;
    ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

    std::vector<Object> objects16;
    generate_proposals(anchors, feat_stride, score_blob, bbox_blob,objects16);

    proposal_boxes.insert(proposal_boxes.end(), objects16.begin(), objects16.end());


    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposal_boxes);


    if (pre_nms_topN > 0 && pre_nms_topN < (int)proposal_boxes.size())
    {
        proposal_boxes.resize(pre_nms_topN);
//        scores.resize(pre_nms_topN);
    }
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposal_boxes, picked, nms_rpn);

    int picked_count = std::min((int)picked.size(), after_nms_topN);

    objects.resize(picked_count);

    std::vector<std::vector<Object> > class_candidates;
    for (int i = 0; i < picked_count; i++)
    {
        objects[i] = proposal_boxes[picked[i]];

        // clip to image size
        float x0 = objects[i].rect.x;
        float y0 = objects[i].rect.y;
        float x1 = x0 + objects[i].rect.width;
        float y1 = y0 + objects[i].rect.height;

        x0 = std::max(std::min(x0, (float)target_size - 1), 0.f);
        y0 = std::max(std::min(y0, (float)target_size - 1), 0.f);
        x1 = std::max(std::min(x1, (float)target_size - 1), 0.f);
        y1 = std::max(std::min(y1, (float)target_size - 1), 0.f);
//        std::cout << x0 << "," << y0 << "," << x1 << "," << y1  << std::endl;
        ncnn::Mat  roi(4);
        roi[0] = x0;
        roi[1] = y0;
        roi[2] = x1;
        roi[3] = y1;
        ncnn::Mat roi_feat;

        psroialign(feat,roi,roi_feat);

        ncnn::Extractor ex2 = thundernet_rcnn.create_extractor();

        ncnn::Mat bbox_pred;
        ncnn::Mat cls_prob;
//        std::cout << roi_feat.c << "," << roi_feat.h << "," << roi_feat.w << std::endl;
        ex2.input("roi_feat", roi_feat);
        ex2.extract("bbox_pred", bbox_pred);
        ex2.extract("cls_score", cls_prob);

        int num_class = cls_prob.w;
        class_candidates.resize(num_class);

        // find class id with highest score
        int label = 0;
        float score = 0.f;
        for (int i = 0; i < num_class; i++)
        {
            float class_score = cls_prob[i];
            if (class_score > score)
            {
                label = i;
                score = class_score;
            }
        }

        // ignore background or low score
        if (label == 20 || score <= confidence_thresh)
            continue;

        //         fprintf(stderr, "%d = %f\n", label, score);

        // unscale to image size
        float rcnn_x1 = roi[0] ;
        float rcnn_y1 = roi[1] ;
        float rcnn_x2 = roi[2] ;
        float rcnn_y2 = roi[3] ;

        float pb_w = rcnn_x2 - rcnn_x1 + 1;
        float pb_h = rcnn_y2 - rcnn_y1 + 1;

        float dx = bbox_pred[label * 4] * 0.1;
        float dy = bbox_pred[label * 4 + 1] * 0.1;
        float dw = bbox_pred[label * 4 + 2] * 0.2;
        float dh = bbox_pred[label * 4 + 3] * 0.2;


        float cx = rcnn_x1 + pb_w * 0.5f;
        float cy = rcnn_y1 + pb_h * 0.5f;

        float obj_cx = cx + pb_w * dx;
        float obj_cy = cy + pb_h * dy;

        float obj_w = pb_w * exp(dw);
        float obj_h = pb_h * exp(dh);

        float obj_x1 = obj_cx - obj_w * 0.5f;
        float obj_y1 = obj_cy - obj_h * 0.5f;
        float obj_x2 = obj_cx + obj_w * 0.5f;
        float obj_y2 = obj_cy + obj_h * 0.5f;

        // clip
        obj_x1 = std::max(std::min(obj_x1, (float)(target_size - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1, (float)(target_size - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2, (float)(target_size - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2, (float)(target_size - 1)), 0.f);

        // append object
        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1/ scale_w, obj_y1/ scale_h, (obj_x2 - obj_x1 + 1)/ scale_w, (obj_y2 - obj_y1 + 1)/ scale_h);
        obj.label = label;
        obj.prob = score;

        class_candidates[label].push_back(obj);


    }

    objects.clear();
    for (int i = 0; i < (int)class_candidates.size(); i++)
    {
        std::vector<Object>& candidates = class_candidates[i];

        qsort_descent_inplace(candidates);

        std::vector<int> picked;
        nms_sorted_bboxes(candidates, picked, nms_threshold);

        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            objects.push_back(candidates[z]);
        }
    }

    qsort_descent_inplace(objects);

    if (max_per_image > 0 && max_per_image < objects.size())
    {
        objects.resize(max_per_image);
    }

    return 0;

}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor","background"
                                       };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imwrite("res.jpg",image);
//    cv::imshow("image", image);
//    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_thundernet(m, objects);

    draw_objects(m, objects);

    return 0;
}
