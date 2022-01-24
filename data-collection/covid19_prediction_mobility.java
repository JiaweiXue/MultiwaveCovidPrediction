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
import java.util.HashMap;
import java.util.HashSet;
import java.util.TimeZone;

import jp.ac.ut.csis.pflow.geom.LonLat;

public class covid19_prediction_mobility{
	/**
	 * codes for read the mobility data
	 * 2021.07.09
	 * @author Taka, Jiawei
	 * @param args
	 */
	protected static final SimpleDateFormat DATE = new SimpleDateFormat("yyyyMMdd");
	protected static final SimpleDateFormat DATETIME = new SimpleDateFormat("yyyyMMdd HH:mm:ss");	
	
	//1. main function
	public static void main(String[] args) throws IOException, ParseException{
		String gpspath = "/mnt/log/covid/loc/";              
		//attention:please change to the location of gps location. ok.
		
		String root = "/mnt/jiawei/";                  
		//attention.ok.
		
		String home = root+"mobility/";                      
		//attention.ok.
		
		String home_loc = home+"id_homelocs.csv";
		String startdate = "20210301"; // start date of mobility data collection
		String enddate   = "20210705"; // end date of mobility data collection
		System.out.println("step 1: start to read the home location csv");  
		//read the potential id (csv)
		HashSet<String> ids_all = new HashSet<String>();
		read_home_loc(home_loc,ids_all);            //read the id_homelocs.csv
		System.out.println("step 2: finish to read the home location csv");  
		
		//initialize the csv files
		Date start_date_date = DATE.parse(startdate);
		Date end_date_date   = DATE.parse(enddate);
		Date date = start_date_date; 
		System.out.println("step 3: start to collect the mobility data");  
		while((date.before(end_date_date))||(date.equals(end_date_date))){
			String date_str = DATE.format(date);
			Date next_date = nextday_date(date);
			System.out.println("step 4: start to read the mobility data .tsv for one day");  
			File gps1 = new File(gpspath+date_str+".tsv"); //read gpspath (tsv)
			if((gps1.exists()) && (gps1.length()>0)){
				File mobility_data = new File(home+date_str+".csv"); //define the written csv
				readTSV(gps1,ids_all,mobility_data);    //write mobility data (csv)
			}
			date = next_date;
			System.out.println("step 4: successfully read the mobility data .tsv for one day"); 
		}
		System.out.println("step 5: finish collecting the mobility data for all days");  
	}	
	
	//2. read the csv file
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
	
	
	//3. read TSV, write csv
	public static void readTSV(
			File in, 
			HashSet<String> ids_all,
			File out)throws NumberFormatException, IOException, ParseException {
		BufferedReader br1 = new BufferedReader(new FileReader(in));
		BufferedWriter bw = new BufferedWriter(new FileWriter(out));
		String line1 = null;
		while((line1 = br1.readLine())!=null){                          //the row is not null
			try {
				String[] tokens = line1.split("\t");
				if(tokens.length>=7){                                 //the row is complete
					String id_br1 = tokens[0];                        //id1 of the user
					if(!id_br1.equals("null")){                       //the id1 is not null 
						if(id_br1.length()>0){                        //the id1's length is larger than 0
							if(tokens[4].length()>=10){               //the time is larger than 10^{9}, Sep 9,2011
							Double lon = Double.parseDouble(tokens[3]);  //read longitude
							Double lat = Double.parseDouble(tokens[2]);  //read latitude
							LonLat p = new LonLat(lon,lat);
							if ((lon>138.8d)&&(lon<140.0d)&&(lat>35.45d)&&(lat<35.95d)){ 
								if(ids_all.contains(id_br1)){
									//the id is in id_homelocs.csv
									
									String unixtime = tokens[4];     // unixtime
									Date currentDate = new Date(Long.parseLong(unixtime)*((long)1000)); // UTC time
									DATETIME.setTimeZone(TimeZone.getTimeZone("GMT+9"));
									String datetime = DATETIME.format(currentDate);
									String date = datetime.split(" ")[0];
									String time = datetime.split(" ")[1];
									Integer hour = Integer.valueOf(time.split(":")[0]);
									Integer min = Integer.valueOf(time.split(":")[1]);
									Integer second = Integer.valueOf(time.split(":")[2]);
									bw.write(id_br1+","+ date+","+
											String.valueOf(hour)+","+
											String.valueOf(min)+","+
											String.valueOf(second)+","+
											String.valueOf(lon)+","+
											String.valueOf(lat));  
									bw.newLine();
								}}}}}}
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
		bw.close();
		br1.close();
	}
	
	//4. next day
	public static Date nextday_date(Date day) throws ParseException{
		Calendar nextCal = Calendar.getInstance();
		nextCal.setTime(day);
		nextCal.add(Calendar.DAY_OF_MONTH, 1);
		Date nextDate = nextCal.getTime();
		return nextDate;
	}
}
