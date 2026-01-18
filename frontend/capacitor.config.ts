import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
    appId: 'com.cloudchaser.app',
    appName: 'CloudChaser',
    webDir: 'out',
    server: {
        // For development with live reload
        // url: 'http://10.0.2.2:3000',
        // cleartext: true
    },
    android: {
        buildOptions: {
            keystorePath: undefined,
            keystorePassword: undefined,
            keystoreAlias: undefined,
            keystoreAliasPassword: undefined,
            releaseType: 'APK',
        }
    },
    plugins: {
        CameraPreview: {
            // Camera preview plugin settings
        }
    }
};

export default config;
