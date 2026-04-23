package com.facerec.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

public class Config {
    private static final Properties props = new Properties();
    private static final String CONFIG_FILE = "config.properties";
    
    static {
        loadConfig();
    }
    
    private static void loadConfig() {
        File configFile = new File(CONFIG_FILE);
        if (configFile.exists()) {
            try (FileInputStream fis = new FileInputStream(configFile)) {
                props.load(fis);
            } catch (IOException e) {
                System.err.println("警告: 无法加载配置文件 " + CONFIG_FILE + ", 使用默认值");
            }
        }
    }
    
    public static String getProperty(String key, String defaultValue) {
        return props.getProperty(key, defaultValue);
    }
    
    public static String getHaarCascadePath() {
        return getProperty("haar.cascade.path", "haarcascade_frontalface_default.xml");
    }
    
    public static String getTrainingDir() {
        String dir = getProperty("training.dir", "data/training");
        new File(dir).mkdirs();
        return dir;
    }
    
    public static String getFaceRegDir() {
        String dir = getProperty("facereg.dir", "data/facereg");
        new File(dir).mkdirs();
        return dir;
    }
    
    public static String getDataDir() {
        String dir = getProperty("data.dir", "data");
        new File(dir).mkdirs();
        return dir;
    }
}
