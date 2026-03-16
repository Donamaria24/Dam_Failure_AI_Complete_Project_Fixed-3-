from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime, timedelta
import json
import uuid
import numpy as np
import math
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced_dam_failure_warning_system'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active data
active_alerts = []
connected_users = []
sos_alerts = []
missing_persons = []
flood_simulation_data = {}

# File paths for data persistence
MISSING_PERSONS_FILE = 'data/missing_persons.json'
SOS_ALERTS_FILE = 'data/sos_alerts.json'

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load existing data on startup
def load_data():
    global missing_persons, sos_alerts
    
    # Load missing persons
    if os.path.exists(MISSING_PERSONS_FILE):
        try:
            with open(MISSING_PERSONS_FILE, 'r') as f:
                missing_persons = json.load(f)
            print(f"✅ Loaded {len(missing_persons)} missing person reports")
        except:
            missing_persons = []
    
    # Load SOS alerts
    if os.path.exists(SOS_ALERTS_FILE):
        try:
            with open(SOS_ALERTS_FILE, 'r') as f:
                sos_alerts = json.load(f)
            print(f"✅ Loaded {len(sos_alerts)} SOS alerts")
        except:
            sos_alerts = []

def save_missing_persons():
    """Save missing persons to JSON file"""
    try:
        with open(MISSING_PERSONS_FILE, 'w') as f:
            json.dump(missing_persons, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving missing persons: {e}")
        return False

def save_sos_alerts():
    """Save SOS alerts to JSON file"""
    try:
        with open(SOS_ALERTS_FILE, 'w') as f:
            json.dump(sos_alerts, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving SOS alerts: {e}")
        return False

@app.route('/')
def index():
    """Enhanced dashboard page with all features"""
    return render_template('index.html')

@app.route('/api/crack-detected', methods=['POST'])
def crack_detected():
    """Enhanced crack detection endpoint with flood simulation"""
    try:
        data = request.get_json()
        
        # Create enhanced alert object
        alert = {
            'id': str(uuid.uuid4()),
            'type': 'crack_detection',
            'severity': 'HIGH',
            'message': 'Critical crack detected in dam structure!',
            'location': data.get('location', 'Mullaperiyar Dam'),
            'confidence': data.get('confidence', 0.85),
            'timestamp': datetime.now().isoformat(),
            'image_data': data.get('image_data', None)
        }
        
        # Store alert
        active_alerts.insert(0, alert)
        if len(active_alerts) > 50:
            active_alerts.pop()
        
        # Trigger flood simulation when crack detected
        flood_prediction = simulate_flood_impact(alert['confidence'])
        alert['flood_prediction'] = flood_prediction
        
        # Broadcast enhanced alert to all connected users
        socketio.emit('crack_alert', alert)
        socketio.emit('flood_update', flood_prediction)
        
        print(f"🚨 ENHANCED CRACK ALERT! Alert ID: {alert['id']}, Confidence: {alert['confidence']}")
        print(f"🌊 Flood simulation triggered: {len(flood_prediction['affected_areas'])} areas analyzed")
        
        return jsonify({
            'status': 'success',
            'message': 'Enhanced alert sent successfully',
            'alert_id': alert['id'],
            'flood_prediction': flood_prediction
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/user-location', methods=['POST'])
def receive_user_location():
    """Enhanced location processing with evacuation planning"""
    try:
        data = request.get_json()
        
        user_location = {
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'timestamp': datetime.now().isoformat(),
            'risk_level': calculate_risk_level(data.get('latitude'), data.get('longitude')),
            'is_simulation': data.get('is_simulation', False)
        }
        
        # Enhanced evacuation info with real-time flood data
        evacuation_info = get_enhanced_evacuation_info(user_location)
        
        return jsonify({
            'status': 'success',
            'location': user_location,
            'evacuation_info': evacuation_info
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/simulate-location', methods=['POST'])
def simulate_location():
    """NEW: Simulate user location for demonstration purposes"""
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lng = data.get('longitude')
        area_name = data.get('area_name', 'Selected Area')
        
        if not lat or not lng:
            return jsonify({
                'status': 'error',
                'message': 'Latitude and longitude are required'
            }), 400
        
        # Create simulated location data
        simulated_location = {
            'latitude': lat,
            'longitude': lng,
            'area_name': area_name,
            'timestamp': datetime.now().isoformat(),
            'risk_level': calculate_risk_level(lat, lng),
            'is_simulation': True
        }
        
        # Get evacuation info for simulated location
        evacuation_info = get_enhanced_evacuation_info(simulated_location)
        
        # Add simulation-specific messaging
        evacuation_info['simulation_note'] = f"📍 SIMULATION MODE: Showing experience for users in {area_name}"
        
        print(f"🎯 Location simulation: {area_name} ({lat}, {lng}) - Risk: {simulated_location['risk_level']}")
        
        return jsonify({
            'status': 'success',
            'location': simulated_location,
            'evacuation_info': evacuation_info
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/emergency-sos', methods=['POST'])
def emergency_sos():
    """Handle SOS emergency alerts with file storage"""
    try:
        data = request.get_json()
        
        sos_alert = {
            'id': str(uuid.uuid4()),
            'type': 'SOS',
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'timestamp': datetime.now().isoformat(),
            'message': data.get('message', 'Emergency assistance needed'),
            'status': 'ACTIVE',
            'contact': data.get('contact', 'Unknown')
        }
        
        sos_alerts.insert(0, sos_alert)
        if len(sos_alerts) > 100:
            sos_alerts.pop()
        
        # Save to file
        if save_sos_alerts():
            print(f"✅ SOS alert saved to file: {sos_alert['id']}")
        
        # Broadcast to all users and rescue coordinators
        socketio.emit('sos_alert', sos_alert)
        
        print(f"🆘 SOS ALERT: {sos_alert['latitude']}, {sos_alert['longitude']}")
        
        return jsonify({'status': 'success', 'sos_id': sos_alert['id']}), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/missing-person', methods=['POST'])
def missing_person_report():
    """FIXED: Handle missing person reports with proper file storage"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('name') or not data.get('age') or not data.get('lastSeen'):
            return jsonify({
                'status': 'error',
                'message': 'Name, age, and last seen location are required fields'
            }), 400
        
        # Create report with auto-generated ID
        report = {
            'id': str(uuid.uuid4()),
            'report_number': f"MP-{len(missing_persons) + 1:04d}",
            'name': data.get('name').strip(),
            'age': int(data.get('age')),
            'lastSeen': data.get('lastSeen').strip(),
            'description': data.get('description', '').strip(),
            'reporterContact': data.get('reporterContact', '').strip(),
            'timestamp': datetime.now().isoformat(),
            'status': 'ACTIVE',
            'reported_by': 'Web User'
        }
        
        # Add to memory
        missing_persons.insert(0, report)
        if len(missing_persons) > 200:
            missing_persons.pop()
        
        # Save to file
        if save_missing_persons():
            print(f"✅ Missing person report saved: {report['name']} (ID: {report['report_number']})")
            
            # Broadcast to rescue coordinators
            socketio.emit('missing_person_report', report)
            
            return jsonify({
                'status': 'success', 
                'message': f'Missing person report submitted successfully!',
                'report_id': report['id'],
                'report_number': report['report_number'],
                'details': {
                    'name': report['name'],
                    'age': report['age'],
                    'total_reports': len(missing_persons)
                }
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to save report to database'
            }), 500
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': 'Invalid age value - must be a number'
        }), 400
    except Exception as e:
        print(f"Error in missing person report: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/flood-analysis-detailed', methods=['GET'])
def get_detailed_flood_analysis():
    """Get comprehensive flood analysis with visualizations"""
    try:
        if not flood_simulation_data:
            return jsonify({
                'status': 'error', 
                'message': 'No flood simulation data available'
            }), 404
        
        # Generate detailed analysis
        detailed_analysis = generate_detailed_flood_analysis()
        
        # Generate visualization charts
        charts = generate_flood_charts()
        
        return jsonify({
            'status': 'success',
            'analysis': detailed_analysis,
            'charts': charts,
            'last_updated': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Analysis error: {str(e)}'
        }), 500

@app.route('/api/trigger-flood-simulation', methods=['POST'])
def trigger_manual_flood_simulation():
    """Manually trigger flood simulation for testing"""
    try:
        data = request.get_json() or {}
        confidence = data.get('confidence', 0.75)
        scenario = data.get('scenario', 'manual_trigger')
        
        # Trigger simulation
        flood_data = simulate_flood_impact(confidence)
        flood_data['trigger_source'] = scenario
        flood_data['manual_trigger'] = True
        
        # Broadcast to all users
        socketio.emit('flood_update', flood_data)
        
        return jsonify({
            'status': 'success',
            'message': 'Flood simulation triggered successfully',
            'simulation_data': flood_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def generate_detailed_flood_analysis():
    """Generate comprehensive flood impact analysis"""
    if not flood_simulation_data or 'affected_areas' not in flood_simulation_data:
        return {}
    
    areas = flood_simulation_data['affected_areas']
    total_risk = flood_simulation_data['total_risk_score']
    
    # Population estimates (approximate)
    population_data = {
        'Vandiperiyar': 15000,
        'Vallakadavu': 8000,
        'Kumily': 25000,
        'Thekkady': 12000
    }
    
    # Calculate impacts
    total_population = sum(population_data.values())
    affected_population = 0
    high_risk_population = 0
    evacuation_zones = []
    economic_impact = 0
    
    detailed_impacts = {}
    
    for area, data in areas.items():
        pop = population_data.get(area, 0)
        risk_level = data['risk_level']
        flood_prob = data['flood_probability']
        
        # Calculate affected population based on flood probability
        area_affected_pop = int(pop * (flood_prob / 100))
        affected_population += area_affected_pop
        
        if risk_level in ['HIGH', 'EXTREME']:
            high_risk_population += area_affected_pop
            evacuation_zones.append({
                'name': area,
                'population': pop,
                'affected_population': area_affected_pop,
                'evacuation_time': data['evacuation_time'],
                'priority': 'IMMEDIATE' if risk_level == 'EXTREME' else 'HIGH'
            })
        
        # Economic impact estimation (in lakhs INR)
        base_impact = {
            'HIGH': 500, 'EXTREME': 1000, 'MEDIUM': 200, 'LOW': 50
        }.get(risk_level, 50)
        
        area_economic_impact = base_impact * (flood_prob / 100)
        economic_impact += area_economic_impact
        
        # Infrastructure analysis
        infrastructure_impact = analyze_infrastructure_impact(area, data)
        
        detailed_impacts[area] = {
            'population': pop,
            'affected_population': area_affected_pop,
            'risk_level': risk_level,
            'flood_probability': flood_prob,
            'water_depth': data['estimated_depth'],
            'evacuation_time': data['evacuation_time'],
            'economic_impact_lakhs': round(area_economic_impact, 2),
            'infrastructure': infrastructure_impact
        }
    
    # Overall analysis
    analysis = {
        'simulation_time': flood_simulation_data['simulation_time'],
        'crack_confidence': flood_simulation_data['crack_confidence'],
        'overall_risk_score': total_risk,
        'summary': {
            'total_population': total_population,
            'affected_population': affected_population,
            'high_risk_population': high_risk_population,
            'affected_percentage': round((affected_population / total_population) * 100, 1),
            'total_economic_impact_lakhs': round(economic_impact, 2),
            'evacuation_zones_count': len(evacuation_zones)
        },
        'evacuation_zones': evacuation_zones,
        'area_details': detailed_impacts,
        'recommendations': generate_recommendations(total_risk, affected_population, evacuation_zones)
    }
    
    return analysis

def analyze_infrastructure_impact(area, flood_data):
    """Analyze infrastructure impact for each area"""
    risk_level = flood_data['risk_level']
    flood_depth = flood_data['estimated_depth']
    
    # Infrastructure categories and their vulnerability
    infrastructure = {
        'roads': {
            'total': {'Vandiperiyar': 25, 'Vallakadavu': 15, 'Kumily': 40, 'Thekkady': 20}.get(area, 20),
            'affected': 0,
            'impact_level': 'LOW'
        },
        'bridges': {
            'total': {'Vandiperiyar': 3, 'Vallakadavu': 2, 'Kumily': 5, 'Thekkady': 2}.get(area, 2),
            'affected': 0,
            'impact_level': 'LOW'
        },
        'schools': {
            'total': {'Vandiperiyar': 8, 'Vallakadavu': 4, 'Kumily': 12, 'Thekkady': 6}.get(area, 5),
            'affected': 0,
            'impact_level': 'LOW'
        },
        'hospitals': {
            'total': {'Vandiperiyar': 2, 'Vallakadavu': 1, 'Kumily': 3, 'Thekkady': 1}.get(area, 1),
            'affected': 0,
            'impact_level': 'LOW'
        },
        'power_stations': {
            'total': {'Vandiperiyar': 1, 'Vallakadavu': 1, 'Kumily': 2, 'Thekkady': 1}.get(area, 1),
            'affected': 0,
            'impact_level': 'LOW'
        }
    }
    
    # Calculate impact based on risk level and flood depth
    impact_multiplier = {
        'EXTREME': 0.9, 'HIGH': 0.7, 'MEDIUM': 0.4, 'LOW': 0.1
    }.get(risk_level, 0.1)
    
    depth_multiplier = min(flood_depth / 2.0, 1.0)  # Cap at 1.0
    final_multiplier = impact_multiplier * depth_multiplier
    
    for infra_type, infra_data in infrastructure.items():
        affected_count = int(infra_data['total'] * final_multiplier)
        infrastructure[infra_type]['affected'] = affected_count
        
        if final_multiplier > 0.7:
            infrastructure[infra_type]['impact_level'] = 'SEVERE'
        elif final_multiplier > 0.4:
            infrastructure[infra_type]['impact_level'] = 'MODERATE'
        elif final_multiplier > 0.1:
            infrastructure[infra_type]['impact_level'] = 'LOW'
    
    return infrastructure

def generate_recommendations(risk_score, affected_population, evacuation_zones):
    """Generate actionable recommendations based on analysis"""
    recommendations = []
    
    if risk_score > 80:
        recommendations.extend([
            "🚨 IMMEDIATE ACTION REQUIRED: Issue evacuation orders for high-risk zones",
            "📢 Activate emergency broadcast systems and sirens",
            "🚁 Deploy rescue teams to evacuation zones immediately"
        ])
    elif risk_score > 60:
        recommendations.extend([
            "⚠️ HIGH ALERT: Prepare evacuation procedures",
            "📱 Send emergency alerts to all residents in affected areas",
            "🚐 Position emergency vehicles at strategic locations"
        ])
    elif risk_score > 40:
        recommendations.extend([
            "🟡 MEDIUM ALERT: Monitor situation closely",
            "📢 Issue advisory warnings to residents",
            "🏥 Ensure emergency services are on standby"
        ])
    
    if affected_population > 20000:
        recommendations.append("🏕️ Establish temporary shelters for large displaced population")
    
    if len(evacuation_zones) > 2:
        recommendations.append("🗺️ Coordinate multi-zone evacuation to prevent traffic congestion")
    
    # Infrastructure specific recommendations
    recommendations.extend([
        "⚡ Secure power stations and backup generators",
        "🌉 Inspect and reinforce critical bridges",
        "🏥 Ensure hospitals have emergency supplies and backup power",
        "📡 Maintain communication systems functionality"
    ])
    
    return recommendations

def generate_flood_charts():
    """Generate visualization charts for flood analysis"""
    if not flood_simulation_data or 'affected_areas' not in flood_simulation_data:
        return {}
    
    charts = {}
    
    try:
        # Chart 1: Risk Level Distribution
        areas = list(flood_simulation_data['affected_areas'].keys())
        risk_levels = [flood_simulation_data['affected_areas'][area]['risk_level'] for area in areas]
        
        plt.figure(figsize=(10, 6))
        risk_counts = {level: risk_levels.count(level) for level in ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']}
        colors = {'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#f97316', 'EXTREME': '#dc2626'}
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(risk_counts.keys(), risk_counts.values(), 
                      color=[colors[level] for level in risk_counts.keys()])
        plt.title('Risk Level Distribution')
        plt.ylabel('Number of Areas')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom')
        
        # Chart 2: Flood Probability by Area
        plt.subplot(1, 2, 2)
        flood_probs = [flood_simulation_data['affected_areas'][area]['flood_probability'] for area in areas]
        bars = plt.bar(areas, flood_probs, color='#3b82f6', alpha=0.7)
        plt.title('Flood Probability by Area')
        plt.ylabel('Flood Probability (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        charts['risk_analysis'] = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()
        
        # Chart 3: Timeline Prediction
        plt.figure(figsize=(12, 6))
        hours = list(range(0, 25, 2))  # 24 hours
        base_risk = flood_simulation_data['total_risk_score']
        
        # Simulate risk progression over time
        risk_progression = []
        for hour in hours:
            if hour < 2:
                risk = base_risk
            elif hour < 6:
                risk = base_risk * 1.2  # Risk increases
            elif hour < 12:
                risk = base_risk * 1.5  # Peak risk
            else:
                risk = base_risk * max(0.3, 1.5 - (hour - 12) * 0.1)  # Risk decreases
            risk_progression.append(min(risk, 100))
        
        plt.plot(hours, risk_progression, 'o-', color='#dc2626', linewidth=3, markersize=6)
        plt.fill_between(hours, risk_progression, alpha=0.3, color='#dc2626')
        plt.title('Predicted Risk Level Over Time')
        plt.xlabel('Hours from Now')
        plt.ylabel('Risk Level (%)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Add critical phases
        plt.axvspan(0, 6, alpha=0.2, color='yellow', label='Critical Phase')
        plt.axvspan(6, 12, alpha=0.2, color='red', label='Peak Danger')
        plt.axvspan(12, 24, alpha=0.2, color='green', label='Recovery')
        plt.legend()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        charts['timeline_prediction'] = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()
        
    except Exception as e:
        print(f"Chart generation error: {e}")
        charts['error'] = f"Chart generation failed: {str(e)}"
    
    return charts

def simulate_flood_impact(crack_confidence):
    """Enhanced flood simulation with more detailed data"""
    global flood_simulation_data
    
    # Simulate affected areas based on crack severity
    base_impact = min(crack_confidence * 100, 95)
    
    # More realistic geographic and demographic data
    affected_areas = {
        'Vandiperiyar': {
            'coordinates': [9.5167, 77.1667],
            'population': 15000,
            'elevation_m': 880,
            'distance_from_dam_km': 8.5,
            'risk_level': 'EXTREME' if base_impact > 80 else 'HIGH' if base_impact > 60 else 'MEDIUM',
            'flood_probability': min(base_impact * 0.9, 95),
            'estimated_depth': base_impact * 0.12,
            'evacuation_time': 20 if base_impact > 80 else 30,
            'critical_infrastructure': ['Hospital', 'Police Station', 'Power Grid']
        },
        'Vallakadavu': {
            'coordinates': [9.5000, 77.2000],
            'population': 8000,
            'elevation_m': 750,
            'distance_from_dam_km': 15.2,
            'risk_level': 'HIGH' if base_impact > 70 else 'MEDIUM' if base_impact > 40 else 'LOW',
            'flood_probability': min(base_impact * 0.75, 85),
            'estimated_depth': base_impact * 0.09,
            'evacuation_time': 35 if base_impact > 70 else 60,
            'critical_infrastructure': ['Bridge', 'School', 'Telecom Tower']
        },
        'Kumily': {
            'coordinates': [9.5947, 77.1636],
            'population': 25000,
            'elevation_m': 950,
            'distance_from_dam_km': 12.0,
            'risk_level': 'MEDIUM' if base_impact > 60 else 'LOW',
            'flood_probability': min(base_impact * 0.4, 65),
            'estimated_depth': base_impact * 0.06,
            'evacuation_time': 90,
            'critical_infrastructure': ['Regional Hospital', 'Bus Terminal', 'Market']
        },
        'Thekkady': {
            'coordinates': [9.5892, 77.1603],
            'population': 12000,
            'elevation_m': 1000,
            'distance_from_dam_km': 10.5,
            'risk_level': 'MEDIUM' if base_impact > 50 else 'LOW',
            'flood_probability': min(base_impact * 0.3, 50),
            'estimated_depth': base_impact * 0.04,
            'evacuation_time': 120,
            'critical_infrastructure': ['Tourist Center', 'Forest Office', 'Wildlife Sanctuary']
        }
    }
    
    # Enhanced flood simulation data
    flood_simulation_data = {
        'simulation_id': str(uuid.uuid4()),
        'crack_confidence': crack_confidence,
        'simulation_time': datetime.now().isoformat(),
        'affected_areas': affected_areas,
        'total_risk_score': base_impact,
        'total_affected_population': sum(area['population'] for area in affected_areas.values()),
        'high_risk_population': sum(area['population'] for area in affected_areas.values() if area['risk_level'] in ['HIGH', 'EXTREME']),
        'estimated_economic_impact_crores': round(base_impact * 2.5, 2),
        'weather_conditions': {
            'rainfall_mm': base_impact * 0.5,
            'wind_speed_kmh': 15 + (base_impact * 0.2),
            'temperature_c': 28
        },
        'dam_status': {
            'water_level_percentage': 85 + (crack_confidence * 15),
            'structural_integrity': 100 - (crack_confidence * 100),
            'spillway_status': 'OPEN' if base_impact > 70 else 'MONITORED'
        }
    }
    
    return flood_simulation_data

@app.route('/api/missing-persons-list', methods=['GET'])
def get_missing_persons_list():
    """Get list of all missing persons reports"""
    try:
        return jsonify({
            'status': 'success',
            'total_reports': len(missing_persons),
            'reports': missing_persons[:50]  # Return last 50 reports
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/rescue-dashboard')
def rescue_dashboard():
    """API for rescue coordinators dashboard"""
    return jsonify({
        'sos_alerts': sos_alerts[:20],
        'missing_persons': missing_persons[:50],
        'crack_alerts': active_alerts[:10],
        'flood_simulation': flood_simulation_data,
        'total_active_alerts': len(active_alerts),
        'total_sos': len(sos_alerts),
        'total_missing': len(missing_persons)
    })

@app.route('/api/system-stats')
def system_stats():
    """Get system statistics"""
    return jsonify({
        'connected_users': len(connected_users),
        'total_alerts': len(active_alerts),
        'sos_alerts': len(sos_alerts),
        'missing_persons': len(missing_persons),
        'system_status': 'OPERATIONAL',
        'last_update': datetime.now().isoformat()
    })

def calculate_risk_level(lat, lng):
    """Enhanced risk calculation with flood simulation data"""
    dam_lat, dam_lng = 9.5419, 77.1539
    
    if lat and lng:
        # Calculate distance using haversine formula (more accurate)
        distance_km = haversine_distance(lat, lng, dam_lat, dam_lng)
        
        if distance_km < 5:  # Within 5km
            return 'EXTREME'
        elif distance_km < 15:  # 5-15km downstream
            return 'HIGH'
        elif distance_km < 30:  # 15-30km 
            return 'MEDIUM'
        else:
            return 'LOW'
    
    return 'UNKNOWN'

def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two points on Earth"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)
    
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def get_enhanced_evacuation_info(location):
    """Enhanced evacuation information with real-time updates"""
    risk_level = location['risk_level']
    
    evacuation_plans = {
        'EXTREME': {
            'action': 'EVACUATE IMMEDIATELY',
            'shelter': 'Kumily Emergency Shelter (Capacity: 2000)',
            'route': 'Take Highway 183 North → Kumily',
            'eta': '30-45 minutes',
            'message': '🔴 EXTREME DANGER: You are very close to dam. Evacuate NOW!',
            'emergency_contacts': ['108 (Ambulance)', '112 (Emergency)', '1077 (Disaster)'],
            'alternate_routes': ['Via Periyar Tiger Reserve', 'Via Mundakayam']
        },
        'HIGH': {
            'action': 'PREPARE TO EVACUATE',
            'shelter': 'Vandiperiyar Community Center (Capacity: 1500)',
            'route': 'Head to higher ground via State Highway 220',
            'eta': '20-30 minutes',
            'message': '🟠 HIGH RISK: Downstream area. Be ready to evacuate immediately.',
            'emergency_contacts': ['108 (Ambulance)', '112 (Emergency)'],
            'alternate_routes': ['Via Kattappana', 'Via Vandanmedu']
        },
        'MEDIUM': {
            'action': 'STAY ALERT',
            'shelter': 'Nearest Community Center',
            'route': 'Monitor alerts and be ready to move',
            'eta': '15-20 minutes to safety',
            'message': '🟡 MEDIUM RISK: Stay informed and monitor updates.',
            'emergency_contacts': ['108 (Ambulance)'],
            'alternate_routes': ['Multiple routes available']
        },
        'LOW': {
            'action': 'MONITOR SITUATION',
            'shelter': 'Current location relatively safe',
            'route': 'No immediate action required',
            'eta': 'N/A',
            'message': '🟢 LOW RISK: Continue monitoring official alerts.',
            'emergency_contacts': ['108 (Emergency)'],
            'alternate_routes': ['Standard evacuation routes']
        }
    }
    
    plan = evacuation_plans.get(risk_level, evacuation_plans['LOW'])
    
    # Add real-time flood data if available
    if flood_simulation_data and 'affected_areas' in flood_simulation_data:
        plan['flood_warning'] = "⚠️ Flood simulation active - Check flood zones before traveling"
    
    return plan

@socketio.on('connect')
def handle_connect():
    """Handle user connection with enhanced data"""
    print(f"👤 User connected: {request.sid}")
    connected_users.append(request.sid)
    
    # Send comprehensive initial data
    emit('recent_alerts', {
        'crack_alerts': active_alerts[:5],
        'sos_count': len(sos_alerts),
        'missing_count': len(missing_persons),
        'system_stats': {
            'connected_users': len(connected_users),
            'system_status': 'OPERATIONAL'
        },
        'flood_data': flood_simulation_data
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle user disconnection"""
    print(f"👤 User disconnected: {request.sid}")
    if request.sid in connected_users:
        connected_users.remove(request.sid)

if __name__ == '__main__':
    # Load existing data on startup
    load_data()
    
    print("🚀 Enhanced Dam Failure Warning System Starting...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔌 API Endpoints:")
    print("   - Crack Detection: /api/crack-detected")
    print("   - Emergency SOS: /api/emergency-sos")  
    print("   - Missing Persons: /api/missing-person")
    print("   - Location Simulator: /api/simulate-location")
    print("   - Flood Analysis: /api/flood-analysis-detailed")
    print("   - Manual Flood Simulation: /api/trigger-flood-simulation")
    print("   - Rescue Dashboard: /api/rescue-dashboard")
    print(f"💾 Data Storage:")
    print(f"   - Missing Persons: {MISSING_PERSONS_FILE}")
    print(f"   - SOS Alerts: {SOS_ALERTS_FILE}")
    print(f"📋 Existing Data:")
    print(f"   - {len(missing_persons)} missing person reports")
    print(f"   - {len(sos_alerts)} SOS alerts")
    print("🆘 Emergency features enabled")
    print("🌊 Advanced flood simulation with charts")
    print("🎯 Location simulator enabled")
    print("📈 Comprehensive impact analysis ready")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
