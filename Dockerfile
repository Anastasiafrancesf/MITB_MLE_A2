# Airflow image for the monthly ML pipeline (data processing, training, inference, monitoring)
FROM apache/airflow:2.10.2-python3.12

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless) for Spark, procps (for 'ps') and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Stable JAVA_HOME path regardless of CPU architecture (amd64 / arm64)
    ln -s /usr/lib/jvm/java-17-openjdk-* /usr/lib/jvm/java-17

ENV JAVA_HOME=/usr/lib/jvm/java-17
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
