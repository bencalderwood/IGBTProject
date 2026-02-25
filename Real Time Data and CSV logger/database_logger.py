import MySQLdb

db = MySQLdb.connect(
    host="localhost",
    user="Dylan",
    password="fredstar321@",
    db="All_Sensors"
)
cursor = db.cursor()

def log_features_to_db(features):
    cursor.execute(
        """
        INSERT INTO HEALTH_FEATURES
        (time, temp_mean, vib_rms, vce_mean, ic_mean, health_state (can remove if not needed))
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            features["timestamp"],
            features["temp_mean"],
            features["vibration_rms"],
            features["vce_mean"],
            features["ic_mean"],
            features["HealthState"]
        )
    )
    db.commit()