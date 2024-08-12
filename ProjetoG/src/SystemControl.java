import java.awt.AWTException;
import java.awt.Robot;

public class SystemControl {
    private Robot robot;

    // Códigos de tecla para aumentar e diminuir volume no Windows
    private static final int VK_VOLUME_UP = 0xAF;     // Código para aumentar volume
    private static final int VK_VOLUME_DOWN = 0xAE;   // Código para diminuir volume

    public SystemControl() throws AWTException {
        this.robot = new Robot();
    }

    public void increaseVolume() {
        robot.keyPress(VK_VOLUME_UP);
        robot.keyRelease(VK_VOLUME_UP);
    }

    public void decreaseVolume() {
        robot.keyPress(VK_VOLUME_DOWN);
        robot.keyRelease(VK_VOLUME_DOWN);
    }
}
