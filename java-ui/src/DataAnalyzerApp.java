import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ItemEvent;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;

public class DataAnalyzerApp extends JFrame {
    private final List<String> pythonPaths = new ArrayList<>();
    private int currentPythonIndex = 0;

    private JLabel lblPythonStatus;
    private JLabel lblSelectedFile;
    private JTextArea txtConsole;
    private JTextArea txtOverview;
    private JTextArea txtEda;
    private JTextArea txtPreprocessing;
    private JTextArea txtModels;
    private JTextArea txtFeatures;
    private JTextArea txtFinal;
    private JTable tblModels;
    private JTable tblPredictions;
    private JLabel lblGraph;
    private JLabel lblHistograms;
    private JLabel lblCorrelationMatrix;
    private JLabel lblLogNormalization;
    private JLabel lblHourNormalization;
    private JComboBox<String> cmbGraphs;

    private File selectedCsvFile;
    private File outputDir = new File("output");

    public DataAnalyzerApp() {
        setTitle("Учебный проект: определение мошеннических банковских операций");
        setSize(1200, 800);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout(8, 8));

        initPythonPaths();
        buildTopPanel();
        buildTabs();
        checkCurrentPython(false);
    }

    private void buildTopPanel() {
        JPanel root = new JPanel(new GridLayout(3, 1, 4, 4));
        root.setBorder(BorderFactory.createEmptyBorder(8, 8, 4, 8));

        JPanel pythonPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton btnCheckPython = new JButton("Проверить Python");
        JButton btnNextPython = new JButton("Следующий Python");
        lblPythonStatus = new JLabel("Python: " + getCurrentPythonPath());
        btnCheckPython.addActionListener(e -> checkCurrentPython(true));
        btnNextPython.addActionListener(e -> switchToNextPython());
        btnNextPython.setEnabled(pythonPaths.size() > 1);
        pythonPanel.add(btnCheckPython);
        pythonPanel.add(btnNextPython);
        pythonPanel.add(lblPythonStatus);

        JPanel filePanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton btnSelectFile = new JButton("1. Выбрать CSV датасет");
        JButton btnRun = new JButton("2. Запустить полный анализ");
        JButton btnOpenOutput = new JButton("Открыть папку результатов");
        lblSelectedFile = new JLabel("Файл не выбран");
        btnSelectFile.addActionListener(e -> selectCsvFile());
        btnRun.addActionListener(e -> runAnalysis());
        btnOpenOutput.addActionListener(e -> openOutputFolder());
        filePanel.add(btnSelectFile);
        filePanel.add(btnRun);
        filePanel.add(btnOpenOutput);
        filePanel.add(lblSelectedFile);

        JLabel hint = new JLabel("Интерфейс запускает Python-скрипт, а затем показывает результаты по вкладкам: обзор, EDA, подготовка, модели, таблицы и графики.");
        hint.setBorder(BorderFactory.createEmptyBorder(0, 4, 0, 0));

        root.add(pythonPanel);
        root.add(filePanel);
        root.add(hint);
        add(root, BorderLayout.NORTH);
    }

    private void buildTabs() {
        JTabbedPane tabs = new JTabbedPane();

        txtOverview = createTextArea();
        txtEda = createTextArea();
        txtPreprocessing = createTextArea();
        txtModels = createTextArea();
        txtFeatures = createTextArea();
        txtFinal = createTextArea();
        txtConsole = createTextArea();

        tabs.addTab("Обзор", new JScrollPane(txtOverview));
        tabs.addTab("EDA", new JScrollPane(txtEda));
        tabs.addTab("Подготовка", new JScrollPane(txtPreprocessing));
        tabs.addTab("Модели", new JScrollPane(txtModels));

        lblHistograms = createImageLabel("Гистограммы всех числовых признаков появятся после анализа");
        tabs.addTab("Распределения", new JScrollPane(lblHistograms));

        lblLogNormalization = createImageLabel("График логарифмической нормализации Amount появится после анализа");
        tabs.addTab("Лог-нормализация", new JScrollPane(lblLogNormalization));

        lblHourNormalization = createImageLabel("Нормализованное распределение операций по часам появится после анализа");
        tabs.addTab("Часы, нормализация", new JScrollPane(lblHourNormalization));

        lblCorrelationMatrix = createImageLabel("Матрица корреляций появится после анализа");
        tabs.addTab("Корреляции", new JScrollPane(lblCorrelationMatrix));

        tblModels = new JTable();
        tabs.addTab("Сравнение моделей", new JScrollPane(tblModels));

        tblPredictions = new JTable();
        tabs.addTab("Предсказания", new JScrollPane(tblPredictions));

        JPanel graphPanel = new JPanel(new BorderLayout(5, 5));
        cmbGraphs = new JComboBox<>();
        cmbGraphs.addItemListener(e -> {
            if (e.getStateChange() == ItemEvent.SELECTED) {
                showSelectedGraph();
            }
        });
        lblGraph = new JLabel("Графики появятся после анализа", SwingConstants.CENTER);
        lblGraph.setBorder(BorderFactory.createEtchedBorder());
        graphPanel.add(cmbGraphs, BorderLayout.NORTH);
        graphPanel.add(new JScrollPane(lblGraph), BorderLayout.CENTER);
        tabs.addTab("Графики", graphPanel);

        tabs.addTab("Важность признаков", new JScrollPane(txtFeatures));
        tabs.addTab("Итог", new JScrollPane(txtFinal));
        tabs.addTab("Консоль", new JScrollPane(txtConsole));

        add(tabs, BorderLayout.CENTER);
    }

    private JTextArea createTextArea() {
        JTextArea area = new JTextArea();
        area.setEditable(false);
        area.setLineWrap(true);
        area.setWrapStyleWord(true);
        area.setFont(new Font("Segoe UI", Font.PLAIN, 15));
        area.setText("Результаты появятся после запуска анализа.");
        return area;
    }

    private JLabel createImageLabel(String text) {
        JLabel label = new JLabel(text, SwingConstants.CENTER);
        label.setBorder(BorderFactory.createEtchedBorder());
        return label;
    }

    private String getCurrentPythonPath() {
        if (pythonPaths.isEmpty()) return "python";
        return pythonPaths.get(currentPythonIndex);
    }

    private void switchToNextPython() {
        if (pythonPaths.size() <= 1) return;
        currentPythonIndex = (currentPythonIndex + 1) % pythonPaths.size();
        lblPythonStatus.setText("Python: " + getCurrentPythonPath());
        lblPythonStatus.setForeground(Color.BLACK);
        appendConsole("Выбран Python: " + getCurrentPythonPath());
    }

    private void checkCurrentPython(boolean showInConsole) {
        String path = getCurrentPythonPath();
        try {
            ProcessBuilder pb = new ProcessBuilder(path, "--version");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            String version = readProcessOutput(process);
            int exit = process.waitFor();
            if (exit == 0 && !version.trim().isEmpty()) {
                lblPythonStatus.setText("Python готов: " + version.trim());
                lblPythonStatus.setForeground(new Color(0, 128, 0));
                if (showInConsole) appendConsole("Проверка Python успешна: " + version.trim());
            } else {
                throw new RuntimeException("Python вернул код " + exit);
            }
        } catch (Exception ex) {
            lblPythonStatus.setText("Python не запустился: " + path);
            lblPythonStatus.setForeground(Color.RED);
            if (showInConsole) appendConsole("Ошибка проверки Python: " + ex.getMessage());
        }
    }

    private void selectCsvFile() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Выберите creditcard.csv");
        int result = chooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            selectedCsvFile = chooser.getSelectedFile();
            lblSelectedFile.setText("Выбран файл: " + selectedCsvFile.getAbsolutePath());
            appendConsole("Выбран CSV-файл: " + selectedCsvFile.getAbsolutePath());
        }
    }

    private void runAnalysis() {
        if (selectedCsvFile == null) {
            JOptionPane.showMessageDialog(this, "Сначала выберите CSV-файл датасета.", "Нет файла", JOptionPane.WARNING_MESSAGE);
            return;
        }

        File script = findPythonScript();
        if (script == null) {
            JOptionPane.showMessageDialog(this,
                    "Не найден файл python/antifraud_pipeline.py.\nПоложите папку python рядом с java-ui или рядом с местом запуска программы.",
                    "Не найден Python-скрипт", JOptionPane.ERROR_MESSAGE);
            return;
        }

        outputDir = new File("output").getAbsoluteFile();
        clearResultsBeforeRun();
        appendConsole("Запуск анализа...");
        appendConsole("Python-скрипт: " + script.getAbsolutePath());
        appendConsole("Датасет: " + selectedCsvFile.getAbsolutePath());
        appendConsole("Папка результатов: " + outputDir.getAbsolutePath());

        new Thread(() -> executePythonScript(script)).start();
    }

    private void executePythonScript(File script) {
        try {
            List<String> command = new ArrayList<>();
            command.add(getCurrentPythonPath());
            command.add(script.getAbsolutePath());
            command.add("--input");
            command.add(selectedCsvFile.getAbsolutePath());
            command.add("--output");
            command.add(outputDir.getAbsolutePath());

            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = reader.readLine()) != null) {
                String text = line;
                SwingUtilities.invokeLater(() -> appendConsole(text));
            }

            int exitCode = process.waitFor();
            SwingUtilities.invokeLater(() -> {
                appendConsole("Python завершился с кодом: " + exitCode);
                if (exitCode == 0) {
                    loadResultsFromOutput();
                    JOptionPane.showMessageDialog(this, "Анализ завершен. Результаты загружены во вкладки.", "Готово", JOptionPane.INFORMATION_MESSAGE);
                } else {
                    JOptionPane.showMessageDialog(this, "Python-скрипт завершился с ошибкой. Подробности во вкладке 'Консоль'.", "Ошибка", JOptionPane.ERROR_MESSAGE);
                }
            });
        } catch (Exception ex) {
            SwingUtilities.invokeLater(() -> {
                appendConsole("Ошибка запуска анализа: " + ex.getMessage());
                JOptionPane.showMessageDialog(this, ex.getMessage(), "Ошибка", JOptionPane.ERROR_MESSAGE);
            });
        }
    }

    private void loadResultsFromOutput() {
        txtOverview.setText(readText(new File(outputDir, "texts/01_summary.txt")));
        txtEda.setText(readText(new File(outputDir, "texts/02_eda_interpretation.txt")));
        txtPreprocessing.setText(readText(new File(outputDir, "texts/03_preprocessing.txt")));
        txtModels.setText(readText(new File(outputDir, "texts/04_models_interpretation.txt")));
        txtFeatures.setText(readText(new File(outputDir, "texts/05_feature_importance.txt")));
        txtFinal.setText(readText(new File(outputDir, "texts/06_final_conclusion.txt")));

        showImageInLabel(lblHistograms, new File(outputDir, "plots/02_all_numeric_histograms.png"), 1000, 620);
        showImageInLabel(lblLogNormalization, new File(outputDir, "plots/04_amount_log_normalization.png"), 1000, 620);
        showImageInLabel(lblHourNormalization, new File(outputDir, "plots/06b_time_distribution_normalized.png"), 1000, 620);
        showImageInLabel(lblCorrelationMatrix, new File(outputDir, "plots/07_full_correlation_heatmap.png"), 1000, 620);

        loadCsvToTable(new File(outputDir, "tables/model_comparison.csv"), tblModels, 200);
        loadCsvToTable(new File(outputDir, "tables/fraud_predictions.csv"), tblPredictions, 100);
        loadGraphs();
    }

    private void clearResultsBeforeRun() {
        txtOverview.setText("Идет анализ...");
        txtEda.setText("Идет анализ...");
        txtPreprocessing.setText("Идет анализ...");
        txtModels.setText("Идет обучение моделей...");
        txtFeatures.setText("Идет расчет важности признаков...");
        txtFinal.setText("Итог будет сформирован после завершения анализа.");
        tblModels.setModel(new DefaultTableModel());
        tblPredictions.setModel(new DefaultTableModel());
        cmbGraphs.removeAllItems();
        lblGraph.setIcon(null);
        lblGraph.setText("Графики появятся после анализа");
        if (lblHistograms != null) { lblHistograms.setIcon(null); lblHistograms.setText("Гистограммы всех числовых признаков появятся после анализа"); }
        if (lblLogNormalization != null) { lblLogNormalization.setIcon(null); lblLogNormalization.setText("График логарифмической нормализации Amount появится после анализа"); }
        if (lblHourNormalization != null) { lblHourNormalization.setIcon(null); lblHourNormalization.setText("Нормализованное распределение операций по часам появится после анализа"); }
        if (lblCorrelationMatrix != null) { lblCorrelationMatrix.setIcon(null); lblCorrelationMatrix.setText("Матрица корреляций появится после анализа"); }
    }

    private File findPythonScript() {
        String[] candidates = {
                "python/antifraud_pipeline.py",
                "../python/antifraud_pipeline.py",
                "../../python/antifraud_pipeline.py",
                "antifraud_pipeline.py"
        };
        for (String c : candidates) {
            File f = new File(c);
            if (f.exists()) return f.getAbsoluteFile();
        }
        return null;
    }

    private String readText(File file) {
        try {
            if (!file.exists()) return "Файл результата не найден: " + file.getAbsolutePath();
            return new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
        } catch (IOException ex) {
            return "Ошибка чтения файла: " + ex.getMessage();
        }
    }

    private void loadCsvToTable(File csvFile, JTable table, int maxRows) {
        if (!csvFile.exists()) {
            table.setModel(new DefaultTableModel(new Object[][]{{"Файл не найден", csvFile.getAbsolutePath()}}, new String[]{"Сообщение", "Путь"}));
            return;
        }

        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(csvFile), StandardCharsets.UTF_8))) {
            String headerLine = br.readLine();
            if (headerLine == null) return;
            headerLine = removeBom(headerLine);
            String[] headers = headerLine.split(",", -1);
            DefaultTableModel model = new DefaultTableModel(headers, 0);

            String line;
            int count = 0;
            while ((line = br.readLine()) != null && count < maxRows) {
                model.addRow(line.split(",", -1));
                count++;
            }
            table.setModel(model);
        } catch (Exception ex) {
            table.setModel(new DefaultTableModel(new Object[][]{{"Ошибка чтения CSV", ex.getMessage()}}, new String[]{"Сообщение", "Ошибка"}));
        }
    }

    private String removeBom(String s) {
        if (s != null && s.startsWith("\uFEFF")) return s.substring(1);
        return s;
    }

    private void loadGraphs() {
        cmbGraphs.removeAllItems();
        File plotsDir = new File(outputDir, "plots");
        File[] files = plotsDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));
        if (files == null || files.length == 0) {
            lblGraph.setText("Графики не найдены.");
            return;
        }
        java.util.Arrays.sort(files, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
        for (File f : files) cmbGraphs.addItem(f.getName());
        if (cmbGraphs.getItemCount() > 0) cmbGraphs.setSelectedIndex(0);
        showSelectedGraph();
    }

    private void showSelectedGraph() {
        if (cmbGraphs.getSelectedItem() == null) return;
        File imageFile = new File(new File(outputDir, "plots"), cmbGraphs.getSelectedItem().toString());
        if (!imageFile.exists()) {
            lblGraph.setIcon(null);
            lblGraph.setText("Файл графика не найден: " + imageFile.getAbsolutePath());
            return;
        }
        showImageInLabel(lblGraph, imageFile, 950, 560);
    }

    private void showImageInLabel(JLabel label, File imageFile, int maxW, int maxH) {
        if (!imageFile.exists()) {
            label.setIcon(null);
            label.setText("Файл графика не найден: " + imageFile.getAbsolutePath());
            return;
        }
        ImageIcon icon = new ImageIcon(imageFile.getAbsolutePath());
        int w = icon.getIconWidth();
        int h = icon.getIconHeight();
        double scale = Math.min((double) maxW / Math.max(w, 1), (double) maxH / Math.max(h, 1));
        if (scale > 1.0) scale = 1.0;
        Image scaled = icon.getImage().getScaledInstance((int) (w * scale), (int) (h * scale), Image.SCALE_SMOOTH);
        label.setIcon(new ImageIcon(scaled));
        label.setText("");
    }

    private void openOutputFolder() {
        try {
            if (!outputDir.exists()) outputDir.mkdirs();
            Desktop.getDesktop().open(outputDir);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "Не удалось открыть папку: " + ex.getMessage(), "Ошибка", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void appendConsole(String text) {
        txtConsole.append(text + "\n");
        txtConsole.setCaretPosition(txtConsole.getDocument().getLength());
    }

    private String readProcessOutput(Process process) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) sb.append(line).append("\n");
        return sb.toString();
    }

    private void initPythonPaths() {
        File localEmbed = new File("python_embed\\python.exe");
        if (localEmbed.exists()) pythonPaths.add(localEmbed.getAbsolutePath());

        try {
            Process p = Runtime.getRuntime().exec(new String[]{"where", "python"});
            BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = in.readLine()) != null) {
                if (!line.trim().isEmpty() && !pythonPaths.contains(line.trim())) pythonPaths.add(line.trim());
            }
        } catch (Exception ignored) {}

        String localAppData = System.getenv("LOCALAPPDATA");
        if (localAppData != null) {
            File pythonProgramsDir = new File(localAppData, "Programs\\Python");
            File[] versions = pythonProgramsDir.listFiles();
            if (versions != null) {
                for (File v : versions) {
                    File exe = new File(v, "python.exe");
                    if (exe.exists() && !pythonPaths.contains(exe.getAbsolutePath())) pythonPaths.add(exe.getAbsolutePath());
                }
            }
        }

        if (!pythonPaths.contains("python")) pythonPaths.add("python");
    }

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {}
        SwingUtilities.invokeLater(() -> new DataAnalyzerApp().setVisible(true));
    }
}
