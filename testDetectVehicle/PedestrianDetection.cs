
using System;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;
using System.Diagnostics;
using Emgu.CV.Util;
#if !(__IOS__ || NETFX_CORE)
using Emgu.CV.Cuda;
#endif

namespace PedestrianDetection
{
    public static class FindPedestrian
    {
        /// <summary>
        /// Find the pedestrian in the image
        /// </summary>
        /// <param name="image">The image</param>
        /// <param name="processingTime">The processing time in milliseconds</param>
        /// <returns>The region where pedestrians are detected</returns>
        public static Rectangle[] Find(IInputArray image, out long processingTime)
        {
            Stopwatch watch;
            Rectangle[] regions;

            using (InputArray iaImage = image.GetInputArray())
            {

                    //this is the CPU/OpenCL version
                    using (HOGDescriptor des = new HOGDescriptor())
                    {
                        des.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());
                        watch = Stopwatch.StartNew();

                        MCvObjectDetection[] results = des.DetectMultiScale(image);
                        regions = new Rectangle[results.Length];
                        for (int i = 0; i < results.Length; i++)
                            regions[i] = results[i].Rect;
                        watch.Stop();
                    }
                }
                processingTime = watch.ElapsedMilliseconds;
                return regions;
        }
    }
}
