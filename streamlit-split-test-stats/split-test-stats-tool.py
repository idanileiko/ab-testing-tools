import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="A/B Testing Analysis",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 A/B Testing Statistical Analysis")
st.write("Upload your experiment data and run statistical tests between groups")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your experiment CSV file", 
    type="csv",
    help="Each row should represent an experiment group from your split test"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded! {df.shape[0]} experiment groups, {df.shape[1]} columns")
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df, use_container_width=True)
        
        # Get column names
        columns = df.columns.tolist()
        
        # Configuration section
        st.subheader("🔧 Analysis Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Population size column selector
            pop_size_column = st.selectbox(
                "Select the population size column:",
                columns,
                help="Column containing the total number of users in each experiment group"
            )
            
            # Group identifier (optional)
            group_id_column = st.selectbox(
                "Select group identifier column (optional):",
                [None] + columns,
                help="Column that identifies each experiment group (e.g., 'control', 'variant_a')"
            )
        
        with col2:
            # Metrics to compare
            metric_columns = st.multiselect(
                "Select metric columns to analyze:",
                [col for col in columns if col != pop_size_column],
                help="Columns containing the metrics you want to compare between groups"
            )
            
            # Significance level
            alpha = st.slider(
                "Significance level (α):",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="Threshold for statistical significance"
            )
        
        if metric_columns:
            st.subheader("📈 Statistical Analysis Results")
            
            # Calculate conversion rates if metrics are counts
            analysis_df = df.copy()
            
            # Add conversion rates for each metric
            for metric in metric_columns:
                rate_col = f"{metric}_rate"
                analysis_df[rate_col] = analysis_df[metric] / analysis_df[pop_size_column]
            
            # Display summary statistics
            st.subheader("📊 Summary Statistics")
            
            summary_data = []
            for i, row in analysis_df.iterrows():
                group_name = row[group_id_column] if group_id_column else f"Group_{i+1}"
                
                for metric in metric_columns:
                    rate = row[metric] / row[pop_size_column]
                    summary_data.append({
                        'Group': group_name,
                        'Metric': metric,
                        'Count': row[metric],
                        'Population': row[pop_size_column],
                        'Rate': rate,
                        'Rate %': f"{rate*100:.2f}%"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Statistical Tests
            st.subheader("🔬 Statistical Test Results")
            
            if len(df) >= 2:
                test_results = []
                
                for metric in metric_columns:
                    st.write(f"**Analysis for: {metric}**")
                    
                    # Prepare data for statistical tests
                    groups_data = []
                    group_names = []
                    
                    for i, row in df.iterrows():
                        group_name = row[group_id_column] if group_id_column else f"Group_{i+1}"
                        successes = int(row[metric])
                        population = int(row[pop_size_column])
                        
                        # Create binary array (1 for success, 0 for failure)
                        group_array = np.concatenate([
                            np.ones(successes),
                            np.zeros(population - successes)
                        ])
                        
                        groups_data.append(group_array)
                        group_names.append(group_name)
                    
                    # Run pairwise comparisons
                    if len(groups_data) == 2:
                        # Two-sample proportion test
                        group1, group2 = groups_data
                        
                        # Calculate proportions
                        p1 = np.mean(group1)
                        p2 = np.mean(group2)
                        n1, n2 = len(group1), len(group2)
                        
                        # Two-proportion z-test
                        pooled_p = (np.sum(group1) + np.sum(group2)) / (n1 + n2)
                        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
                        z_score = (p1 - p2) / se
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        
                        # Effect size (difference in proportions)
                        effect_size = abs(p1 - p2)
                        
                        # Confidence interval for difference
                        se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
                        margin_error = stats.norm.ppf(1-alpha/2) * se_diff
                        ci_lower = (p1 - p2) - margin_error
                        ci_upper = (p1 - p2) + margin_error
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                f"{group_names[0]} Rate", 
                                f"{p1:.4f} ({p1*100:.2f}%)"
                            )
                        with col2:
                            st.metric(
                                f"{group_names[1]} Rate", 
                                f"{p2:.4f} ({p2*100:.2f}%)"
                            )
                        with col3:
                            significance = "✅ Significant" if p_value < alpha else "❌ Not Significant"
                            st.metric("P-value", f"{p_value:.6f}")
                            st.write(significance)
                        
                        # Detailed results
                        results_data = {
                            "Metric": [metric],
                            "Z-score": [f"{z_score:.4f}"],
                            "P-value": [f"{p_value:.6f}"],
                            "Significant": [p_value < alpha],
                            "Effect Size": [f"{effect_size:.4f}"],
                            "95% CI": [f"({ci_lower:.4f}, {ci_upper:.4f})"]
                        }
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                    elif len(groups_data) > 2:
                        # Multiple groups - Chi-square test
                        st.write("**Multi-group Analysis (Chi-square test)**")
                        
                        # Prepare contingency table
                        successes = [np.sum(group) for group in groups_data]
                        failures = [len(group) - np.sum(group) for group in groups_data]
                        
                        contingency_table = np.array([successes, failures])
                        
                        # Chi-square test
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Chi-square", f"{chi2:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.6f}")
                        with col3:
                            significance = "✅ Significant" if p_value < alpha else "❌ Not Significant"
                            st.write(significance)
                        
                        # Show group rates
                        rates_data = []
                        for i, (group_data, group_name) in enumerate(zip(groups_data, group_names)):
                            rate = np.mean(group_data)
                            rates_data.append({
                                'Group': group_name,
                                'Rate': f"{rate:.4f}",
                                'Rate %': f"{rate*100:.2f}%",
                                'Count': f"{int(np.sum(group_data))}/{len(group_data)}"
                            })
                        
                        rates_df = pd.DataFrame(rates_data)
                        st.dataframe(rates_df, use_container_width=True)
                    
                    # Visualization
                    st.subheader("📈 Visualization")
                    
                    # Create visualization data
                    viz_data = []
                    for i, (group_data, group_name) in enumerate(zip(groups_data, group_names)):
                        rate = np.mean(group_data)
                        population = len(group_data)
                        count = int(np.sum(group_data))
                        
                        viz_data.append({
                            'Group': group_name,
                            'Conversion_Rate': rate,
                            'Count': count,
                            'Population': population
                        })
                    
                    viz_df = pd.DataFrame(viz_data)
                    
                    # Bar chart of conversion rates
                    fig = px.bar(
                        viz_df, 
                        x='Group', 
                        y='Conversion_Rate',
                        title=f'Conversion Rates by Group - {metric}',
                        text='Conversion_Rate'
                    )
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig.update_layout(yaxis_title="Conversion Rate")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
            
            else:
                st.warning("⚠️ Need at least 2 groups to run statistical tests")
        
        # Export results
        if metric_columns and len(df) >= 2:
            st.subheader("💾 Export Results")
            
            # Create downloadable report
            report_data = []
            for metric in metric_columns:
                for i, row in df.iterrows():
                    group_name = row[group_id_column] if group_id_column else f"Group_{i+1}"
                    rate = row[metric] / row[pop_size_column]
                    
                    report_data.append({
                        'Metric': metric,
                        'Group': group_name,
                        'Count': row[metric],
                        'Population': row[pop_size_column],
                        'Conversion_Rate': rate,
                        'Conversion_Rate_Percent': f"{rate*100:.2f}%"
                    })
            
            report_df = pd.DataFrame(report_data)
            csv_report = report_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Analysis Report",
                data=csv_report,
                file_name="ab_test_analysis.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Error processing the file: {str(e)}")
        st.write("Please ensure your CSV file is properly formatted with numeric data.")

else:
    st.info("👆 Upload your experiment CSV file to begin analysis")
    
    # Instructions
    with st.expander("📖 How to use this app"):
        st.markdown("""
        **Data Format Expected:**
        - Each row = one experiment group
        - One column = total population size for that group
        - Other columns = metric counts (conversions, clicks, etc.)
        
        **Example CSV:**
        ```
        group_name,population,conversions,clicks,signups
        control,10000,850,2340,123
        variant_a,10000,920,2180,145
        variant_b,10000,780,2890,98
        ```
        
        **Steps:**
        1. Upload your CSV file
        2. Select the column containing population sizes
        3. Select the metric columns you want to analyze
        4. Review statistical test results
        5. Download analysis report
        
        **Statistical Tests:**
        - 2 groups: Two-proportion Z-test
        - 3+ groups: Chi-square test of independence
        """)
        
    # Sample data generator
    with st.expander("🎯 Generate Sample Data"):
        if st.button("Create Sample A/B Test Data"):
            sample_data = {
                'group': ['control', 'variant_a', 'variant_b'],
                'population': [10000, 10000, 10000],
                'conversions': [850, 920, 780],
                'clicks': [2340, 2180, 2890],
                'signups': [123, 145, 98]
            }
            sample_df = pd.DataFrame(sample_data)
            
            st.write("Sample data generated:")
            st.dataframe(sample_df, use_container_width=True)
            
            csv_sample = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Data",
                data=csv_sample,
                file_name="sample_ab_test_data.csv",
                mime="text/csv"
            )
