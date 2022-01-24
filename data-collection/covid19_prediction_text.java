package covid.prediction.xue_taka;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.Map;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

import jp.ac.ut.csis.pflow.geom.LonLat;

public class covid19_prediction_text{
	/**
	 * codes for mining the web search data
	 * 2021.07.14
	 * @author Jiawei Xue, Taka
	 * @param args
	 */
	protected static final SimpleDateFormat DATE = new SimpleDateFormat("yyyyMMdd");
	protected static final SimpleDateFormat DATETIME = new SimpleDateFormat("yyyyMMdd HH:mm:ss");
	
	//1. main function
	public static void main(String[] args) throws IOException, ParseException{ 
		//to read the text data
		//String textpath = "/mnt/home1/q_emotion/q_sum/";   
		String textpath = "/mnt/log/covid/";              //added: 20210423
		//attention: please change it to the location of raw web search data!
		
		//to read id_homelocs.csv
		String root = "/mnt/jiawei/";   
		String home = root+"mobility/"; 
		String home_loc = home+"id_homelocs.csv";  
		//attention: to read id_homelocs.csv, which is the output of "covid19_prediction_home.java"
		
		//to read the given 130-dimension web search text
		//to store the the frequency of web search record
		
		String home_text = root+"text/"; 
		File home_text_f = new File(home_text); home_text_f.mkdir();
		String symptom_loc = home_text + "symptom_20210310.csv";   
		//attention: please change to potential symptom csv file
		System.out.println("step 1: successively define the name of symptom files");  
		System.out.println("step 1: please remember to add Taka's symptom files to your server"); 
		
		String startdate = "20210601"; // start date to store the text data
		String enddate   = "20210620"; // end date to store the text data
		
		//step 1: read the id_homelocs.csv
		HashSet<String> ids_all = new HashSet<String>();
		read_home_loc(home_loc,ids_all);               
		System.out.println("step 2: successively read the home location files"); 
		System.out.println("the total number of potential Tokyo residents is: "); //added:20210415
		System.out.println(ids_all.size());   //added:20210415
		
		//step 2: read the 130-dimension symptom
		HashMap<String, Integer> symptom_idx = new HashMap<String, Integer>();
		read_symptom(symptom_loc,symptom_idx);  
		System.out.println("step 3: successively read the symptom files, which is"); 
		System.out.println(symptom_idx); //added:20210414
		
		//step 3: generate the search frequency count everyday
		Date start_date_date = DATE.parse(startdate);
		Date end_date_date   = DATE.parse(enddate);
		Date date = start_date_date; 
		System.out.println("step 4: start to read tsv file"); 
		while((date.before(end_date_date))||(date.equals(end_date_date))){
			String date_str = DATE.format(date);
			Date next_date = nextday_date(date);
			System.out.println("step 4: start to read tsv file for "+date_str); 
			//File text1 = new File(textpath+date_str+"_japan.tsv");   
			File text1 = new File(textpath + "/query/" + date_str + ".tsv");  //added: 20210423, 20210426  
			//attention: please change to the raw web search file, ok.
			//20210709: add / before "query".
			
			if((text1.exists()) && (text1.length()>0)){
				System.out.println("the number of daily original records is: "); 
				System.out.println(text1.length());   //added:20210415
				
				File text_data = new File(home_text+date_str+"_text"+".csv"); //define the written csv
				readTSV(text1,ids_all,symptom_idx,text_data);    //write text_data
			}
			date = next_date;
			System.out.println("step 4: end reading tsv file for "+date_str); 
		}
		System.out.println("step 5: finish the reading all tsv"); 
	}
	
	//2. read the csv file, which is the same as the function in "covid19_prediction_mobility.java"
	public static void read_home_loc(
			String home_loc,          //input file                                                
			HashSet<String> ids)throws NumberFormatException, IOException, ParseException {     
		//potential user
		File hom_loc_file = new File(home_loc); 
		BufferedReader loc = new BufferedReader(new FileReader(hom_loc_file));
		String line1 = null;
		while((line1 = loc.readLine())!=null){          //the row is not null
			try{
				String[] tokens = line1.split(",");
				String id_br1 = tokens[0];  
				ids.add(id_br1);                    //add the id into ids
			}
			catch (Exception  e){
				System.out.println("OTHER ERROR IN LINE ----");
				System.out.println(line1);
				System.out.println("----");				
			}
		}
		loc.close();
	}
	
	//3. read the symptom csv, which is a predefined potential symptom list
	public static void read_symptom(
			String symptom_loc,                    //read the symptom csv   
			HashMap<String, Integer> symptom_idx   //output symptom_idx
			)throws NumberFormatException, IOException, ParseException{
		File sym_loc_file = new File(symptom_loc); 
		BufferedReader sym = new BufferedReader(new FileReader(sym_loc_file));
		String line1 = null;
		Integer num = 0;
		while((line1 = sym.readLine())!=null){  //the row is not null
			try{
				String[] tokens = line1.split("\t");
				String id_br1 = tokens[0];     //each row is a symptom
				symptom_idx.put(id_br1,num);     //HashMap["cough"] = 10, because fever is the 11-the element in the list 
				num += 1;                              
			}
			catch (Exception  e){
				System.out.println("OTHER ERROR IN LINE ----");
				System.out.println(line1);
				System.out.println("----");				
			}
		}
		sym.close();	
	}
	
	//4. read the text data
	public static void readTSV(
			File in,                                  //the web search record 
			HashSet<String> ids_all,                  //the Tokyo residence id
			HashMap<String, Integer> symptom_idx,     //the symptom file
			File out)throws NumberFormatException, IOException, ParseException{                                //the output all
		//frequency_all[id] = {"cough":3,"fever":5,...}
		HashMap<String, HashMap<String,Integer>> frequency_all = new HashMap<String, HashMap<String,Integer>>();
		BufferedReader br1 = new BufferedReader(new FileReader(in));
		BufferedWriter bw = new BufferedWriter(new FileWriter(out));
		String line1 = null;
		Integer n1 = 0; //added: 20210415
		while((line1 = br1.readLine())!=null){                          //the row is not null
			try {
				//String[] tokens  = line1.split("\u0001");//added:20210420
				String[] tokens = line1.split("\t");       //added:20210423
				if(tokens.length>=3){                                 //the row is complete
					String id_br1 = tokens[1];                        //id1 of the user
					if(!id_br1.equals("null")){                       //the id1 is not null 
						if(id_br1.length()>0){                        //the id1's length is larger than 0
							if(ids_all.contains(id_br1)){             //the id is in id_homelocs.csv
								//read the query text
								n1 +=1 ; //added: 20210415
								String raw_text_data = tokens[3]; 
								HashMap<String,Integer> count_frequency = new HashMap<String,Integer>();
								if (frequency_all.keySet().contains(id_br1)){
									count_frequency = frequency_all.get(id_br1);  //get the frequency record of id_br1 in frequency_all
								}
								refresh_frequency(raw_text_data,symptom_idx,count_frequency); //refresh the record using raw_text_data
								frequency_all.put(id_br1, count_frequency); //refresh the frequency_all	
							}
						}
					}
				}
			}
			catch (ArrayIndexOutOfBoundsException  e){
				System.out.println("OUT OF BOUNDS EXCEPTION ----");
				System.out.println(line1);
				System.out.println("----");
			}
			catch (Exception  e){
				System.out.println("OTHER ERROR IN LINE ----");
				System.out.println(line1);
				System.out.println("----");				
			}
		}
		System.out.println("the number of daily records whose users are Tokyo residences"); //added:20210415
		System.out.println(n1);   //added:20210415
		System.out.println("the number of texted users on one day"); //added:20210414
		System.out.println(frequency_all.size());   //added:20210414
		//write the frequencyResult to ouput
		for (String row : frequency_all.keySet()){  //for each id
			HashMap<String,Integer> value = (HashMap<String,Integer>) frequency_all.get(row);
			for (String text1: value.keySet()){   //for each word
				Integer count1 = (Integer) value.get(text1);
				row = row + "," + text1 + "," + String.valueOf(count1);
			}
			bw.write(row);  
			bw.newLine();
	    }		
		bw.close();
		br1.close();
		
	}
	
	//5. use the web search record (text) and all potential symptoms (symptom_idx)
	//to update the web search frequency count (output) for this id.
	public static void refresh_frequency(String text_data, HashMap<String, Integer> symptom_idx, HashMap<String, Integer> text_output){
		for (String symptom : symptom_idx.keySet()){ //for each potential symptom
			if (text_data.contains(symptom)){
				if (!text_output.keySet().contains(symptom)) {
					text_output.put(symptom,1);	
				}
				else {
					text_output.put(symptom,(Integer)text_output.get(symptom)+1);
				}	
			}
	    }
	}
	
	//6. next day
	public static Date nextday_date(Date day) throws ParseException{
		Calendar nextCal = Calendar.getInstance();
		nextCal.setTime(day);
		nextCal.add(Calendar.DAY_OF_MONTH, 1);
		Date nextDate = nextCal.getTime();
		return nextDate;
	}
}
