import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.awt.AWTException;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.WindowConstants;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        try {
            // Inicializa o controle de sistema
            SystemControl systemControl = new SystemControl();

            // Inicializa a captura de vídeo
            VideoCapture camera = new VideoCapture(0);
            if (!camera.isOpened()) {
                System.out.println("Erro ao abrir a câmera!");
                return;
            }

            Mat frame = new Mat();
            Mat processed = new Mat();

            // Cria uma janela para exibir a imagem da câmera
            JFrame window = new JFrame("Camera Feed");
            JLabel imageLabel = new JLabel();
            window.add(imageLabel);
            window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            window.setSize(640, 480);
            window.setVisible(true);

            // Loop para captura e processamento contínuo de frames da câmera
            while (window.isVisible()) {
                if (camera.read(frame)) {
                    // Processa a imagem
                    Imgproc.cvtColor(frame, processed, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.GaussianBlur(processed, processed, new Size(15, 15), 0);
                    Imgproc.threshold(processed, processed, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

                    // Encontra contornos
                    List<MatOfPoint> contours = new ArrayList<>();
                    Mat hierarchy = new Mat();
                    Imgproc.findContours(processed, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

                    // Exemplo de controle baseado em número de contornos
                    if (contours.size() == 2) { // Exemplo de gesto detectado
                        systemControl.increaseVolume();
                    } else if (contours.size() == 3) { // Outro gesto detectado
                        systemControl.decreaseVolume();
                    }

                    // Converte o frame para BufferedImage e exibe na janela
                    BufferedImage img = matToBufferedImage(frame);
                    imageLabel.setIcon(new ImageIcon(img));
                    window.repaint();
                }
            }

            // Libera a câmera
            camera.release();
            window.dispose();

        } catch (AWTException e) {
            e.printStackTrace();
        }
    }

    // Função auxiliar para converter Mat (OpenCV) para BufferedImage (Java)
    public static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] b = new byte[bufferSize];
        mat.get(0, 0, b);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((java.awt.image.DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }
}
