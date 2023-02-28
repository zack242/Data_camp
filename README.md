# Ramp Challenge - Predicting Air Quality

## Authors : 

- Karine ZAFY
- Gauthier MIRALLES
- Anthony NGUYEN
- Zakaria TOZY 
- Angie MENDEZ LLANOS
- Marius Bartel SAMO KAMGA


This is a GitHub repository for a data science challenge.

For this challenge, participants will be using meteorological data from different stations in Beijing, which will be merged to create a comprehensive dataset. This is important because air pollution levels can vary significantly across different areas of a city, depending on factors such as traffic density, industrial activity, and weather patterns. By combining data from multiple stations, we can capture this variation and obtain a more accurate representation of air quality across the city.
The merged dataset will include various meteorological variables such as temperature, humidity, wind speed, and precipitation, as well as PM2.5 concentrations, which will serve as the target variable for the predictive model. Participants will be tasked with developing a model that can accurately predict PM2.5 concentrations based on the available meteorological data, with the ultimate goal of identifying effective strategies for air pollution reduction and promoting public health and environmental sustainability

# Getting Started

## Clone the Repository

```bash
git clone https://github.com/zack242/Data_camp.git
```

## Set Up the Environment

1. Open the cloned directory

```bash
cd Data_camp
```

2. Install the ramp-workflow library

```bash
pip install ramp-workflow
```

Go to the ramp-workflow wiki for more help on the RAMP ecosystem.

3. Install the requirement

```
pip install -r requirements.txt
```

## Download the data

```
python download_data.py
```

## Test the starting kit

```
!ramp-test --submission starting_kit  --quick-test
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
