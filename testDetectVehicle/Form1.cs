using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Cuda;
using PedestrianDetection;
using System.Diagnostics;

namespace testDetectVehicle
{
    public partial class Form1 : Form
    {
        Timer My_Timer = new Timer();
        int FPS = 30;
        List<Image<Bgr, Byte>> image_array = new List<Image<Bgr, Byte>>();
        Capture _capture;
        public Form1()
        {
            InitializeComponent();
            Image<Bgr, byte> image1 = new Image<Bgr, byte>("p1.jpg");
            this.WindowState = FormWindowState.Maximized;
            pictureBox1.Height = this.Height;
            pictureBox1.Width = this.Width;
            My_Timer.Interval = 1000 / FPS;
            My_Timer.Tick += new EventHandler(My_Timer_Tick);
            My_Timer.Start();
            _capture = new Capture("p3.mkv");

        }
        private void ProcessFrame(object sender, EventArgs arg)
        {
            Mat frame = _capture.QueryFrame();
            pictureBox1.Height = frame.Height;
            pictureBox1.Width = frame.Width;
            if (frame != null)
            {
                //image_array.Add(frame.Copy());
                long processingTime;
                Rectangle[] results;

                if (CudaInvoke.HasCuda)
                {
                    using (GpuMat gpuMat = new GpuMat(frame))
                        results = FindPedestrian.Find(gpuMat, out processingTime);
                }
                else
                {
                    using (UMat uImage = frame.ToUMat(AccessType.ReadWrite))
                        results = FindPedestrian.Find(uImage, out processingTime);
                }

                foreach (Rectangle rect in results)
                {
                    CvInvoke.Rectangle(frame, rect, new Bgr(Color.Red).MCvScalar);
                }
                Image<Bgr, Byte> img = frame.ToImage<Bgr, Byte>();
                img.Resize(384, 288, Emgu.CV.CvEnum.Inter.Linear);
                Image<Gray, Byte> grayFrame = img.Convert<Gray, Byte>();
                Image<Gray, Byte> smallGrayFrame = grayFrame.PyrDown();
                Image<Gray, Byte> smoothedGrayFrame = smallGrayFrame.PyrUp();
                pictureBox1.Image = smoothedGrayFrame.ToBitmap();
            }
            else
            {
                Application.Idle -= ProcessFrame;// treat as end of file
            }

        }
        private void My_Timer_Tick(object sender, EventArgs e)
        {
           Mat frame = _capture.QueryFrame();
            pictureBox1.Height = frame.Height;
            pictureBox1.Width = frame.Width;
            if (frame != null)
            {
                long processingTime;
                Rectangle[] results;

                if (CudaInvoke.HasCuda)
                {
                    using (GpuMat gpuMat = new GpuMat(frame))
                        results = FindPedestrian.Find(gpuMat, out processingTime);
                }
                else
                {
                    using (UMat uImage = frame.ToUMat(AccessType.ReadWrite))
                        results = FindPedestrian.Find(uImage, out processingTime);
                }

                foreach (Rectangle rect in results)
                {
                    CvInvoke.Rectangle(frame, rect, new Bgr(Color.Red).MCvScalar);
                }
                Image<Bgr, Byte> img = frame.ToImage<Bgr, Byte>();
                img.Resize(384, 288, Emgu.CV.CvEnum.Inter.Linear);
                Image<Gray, Byte> grayFrame = img.Convert<Gray, Byte>();
                Image<Gray, Byte> smallGrayFrame = grayFrame.PyrDown();
                Image<Gray, Byte> smoothedGrayFrame = smallGrayFrame.PyrUp();
                pictureBox1.Image = smoothedGrayFrame.ToBitmap();
            }
            else
            {
                My_Timer.Stop();
            }
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            
        }
    
        public void viewing()
        {
            Image<Bgr, byte> image1 = new Image<Bgr, byte>("p1.jpg");
            this.WindowState = FormWindowState.Maximized;
            pictureBox1.Height = this.Height;
            pictureBox1.Width = this.Width;
            image1 = image1.Resize(this.Width, this.Height, Emgu.CV.CvEnum.Inter.Linear);
            Mat image = image1.Mat;
            long processingTime;
            Rectangle[] results;

            if (CudaInvoke.HasCuda)
            {
                using (GpuMat gpuMat = new GpuMat(image))
                    results = FindPedestrian.Find(gpuMat, out processingTime);
            }
            else
            {
                using (UMat uImage = image.ToUMat(AccessType.ReadWrite))
                    results = FindPedestrian.Find(uImage, out processingTime);
            }

            foreach (Rectangle rect in results)
            {
                CvInvoke.Rectangle(image, rect, new Bgr(Color.Red).MCvScalar);
            }
            Image<Bgr, Byte> img = image.ToImage<Bgr, Byte>();
            pictureBox1.Image = img.ToBitmap();
        }

    }
}
