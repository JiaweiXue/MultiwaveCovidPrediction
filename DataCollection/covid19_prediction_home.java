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
import java.util.Iterator;
import java.util.TimeZone;

import jp.ac.ut.csis.pflow.geom.LonLat;

public class covid19_prediction_home{
	/**
	 * codes for mining the home-location relationship
	 * 2021.02.28
	 * @author Taka, Jiawei
	 * @param args
	 */
	protected static final SimpleDateFormat DATE = new SimpleDateFormat("yyyyMMdd");
	protected static final SimpleDateFormat DATETIME = new SimpleDateFormat("yyyyMMdd HH:mm:ss");
	
	//1. main function
	public static void main(String[] args) throws IOException, ParseException{
		String gpspath = "/mnt/log/covid/loc/";                                              
		//attention:please change to the location of gps location. ok.
		
		String root = "/mnt/jiawei/"; File root_f = new File(root); root_f.mkdir();    
		//attention:build a new location. ok.
		System.out.println("step 1: successively build the /mnt/jiawei/");  
		
		
		String home = root+"mobility/"; File home_f = new File(home); home_f.mkdir();        
		//attention:build a new location. ok.
		System.out.println("step 2: successively build the /mobility/");  
		
		//we store (1) output id_homelocs.csv; (2) mobility csv every day here.

		String startdate = "20200106"; // start date to detect the home location
		String enddate   = "20200119"; // end date to detect the home location
		
		//estimate and output home using MeanShift 
		File idhome_f = new File(home+"id_homelocs.csv");
		System.out.println("step 3: successively set up \"id_homelocs.csv\"");  
		System.out.println("step 4: successively start to estimate home location");  
		getHomes(startdate,enddate,gpspath,idhome_f);
		System.out.println("step 5: successively finish estimating home location");  
	}
	
	//2. getHomes 
	public static void getHomes(
			String startdate,                      
			String enddate,                            
			String gpspath,                              //gps path
			File idhome                                  //output file
			) throws NumberFormatException, IOException, ParseException {
		HashMap<String, HashMap<String, LonLat>> id_datetime_ll = new HashMap<String, HashMap<String, LonLat>>();
		
		//Xue: load the data day by day
		Date start_date_date = DATE.parse(startdate);
		Date end_date_date   = DATE.parse(enddate);
		Date date = start_date_date;
		while((date.before(end_date_date))||(date.equals(end_date_date))){
			String date_str = DATE.format(date);
			System.out.println("step 4: start collecting the data for " + date_str);  
			Date next_date = nextday_date(date);
			
			//Xue: load the gpspath data on one day	
			File gps1 = new File(gpspath+date_str+".tsv"); //load the gps data on this day
			if((gps1.exists()) && (gps1.length()>0)){
				
				//Xue: store the information in id_datetime_11
				getlogs(gps1, date_str, id_datetime_ll);
			}
			date = next_date;
			System.out.println("step 4: finish collecting the data for " + date_str);  
		}
		System.out.println("step 4: finish collecting all data for all dates"); 
		gethomelocs(id_datetime_ll, idhome); //write the location results
		System.out.println("step 5: finish estimating the home location "); 
	}
	
	
	//3. getLogs for all day
	public static void getlogs(
			File in,                                                    //input file
			String date,                                                  
			HashMap<String, HashMap<String, LonLat>> id_datetime_ll     //output file <id_br1:<datetime, LonLat(lon,lat)>>
			) throws NumberFormatException, IOException, ParseException{
		
		//Xue: use the BufferReader to enable efficient data reading
		BufferedReader br1 = new BufferedReader(new FileReader(in));
		String line1 = null;
		System.out.println("step 4: start getting log for the day");  
		//Xue: read the GPS data line by line
		while((line1=br1.readLine())!=null){                          //the row is not null
			try {
				String[] tokens = line1.split("\t");
				if(tokens.length>=7){                                 //the row is complete
					String id_br1 = tokens[0];                        //id1 of the user
					if(!id_br1.equals("null")){                       //the id1 is not null 
						if(id_br1.length()>0){                        //the id1's length is larger than 0
							if(tokens[4].length()>=10){               //the time is larger than 10^{9}, Sep 9,2011
								
							//Xue: read the longitude, latitude
							Double lon = Double.parseDouble(tokens[3]);  //read longitude
							Double lat = Double.parseDouble(tokens[2]);  //read latitude
							LonLat p = new LonLat(lon,lat);
							if ((lon>139.55d)&&(lon<139.92d)&&(lat>35.52d)&&(lat<35.82d)){  //the point is near Tokyo Prefecture
								//Xue: get the time
								String unixtime = tokens[4];     // unixtime
								Date currentDate = new Date(Long.parseLong(unixtime)*((long)1000)); // UTC time
								DATETIME.setTimeZone(TimeZone.getTimeZone("GMT+9"));
								String datetime = DATETIME.format(currentDate);
								String time = datetime.split(" ")[1];
								Integer hour = Integer.valueOf(time.split(":")[0]);
								if((hour<9)||(hour>18)) {
									if(id_datetime_ll.containsKey(id_br1)){
										//Xue: add the night hour records into the data
										id_datetime_ll.get(id_br1).put(datetime, p);
									}
									else{
										HashMap<String, LonLat> tmp = new HashMap<String, LonLat>();
										tmp.put(datetime, p);
										id_datetime_ll.put(id_br1, tmp);
								}}}}}}}
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
		System.out.println("step 4: finish getting log for the day"); 
		br1.close();
	}
	
	
	//4. get home location
	public static HashMap<String, LonLat> gethomelocs(
			HashMap<String, HashMap<String, LonLat>> id_datetime_ll,
			File out
			)throws IOException{
		HashMap<String, LonLat> ids = new HashMap<String, LonLat>();
		
		//Xue: export the homelocs (id:LonLat)
		//System.out.println("step 5: start estimating the home"); 
		BufferedWriter bw = new BufferedWriter(new FileWriter(out));
		for (String id : id_datetime_ll.keySet()){	
			//Xue: select the data related to an id
			HashMap<String, LonLat> datapoints = id_datetime_ll.get(id);
			if (datapoints.size()>=14){          //the user has >= 14 night records during Jan. 2021
				LonLat home = meanshift(datapoints, 200d, 100d, 500d); //hyperparameters.
				if(home.getLon()!=0d) {
					Double lon = home.getLon();
					Double lat = home.getLat();
					//Xue: update ids
					ids.put(id, new LonLat(lon,lat));
					//Xue: write bw
					bw.write(id+","+
							String.valueOf(ids.get(id).getLon())+","+
							String.valueOf(ids.get(id).getLat())); 
					bw.newLine();
				}
			}
		}
		//System.out.println("step 5: finish estimating the home"); 
		bw.close();
		return ids;
	}
	
	//5. meanshift function
	public static LonLat meanshift(
			HashMap<String, LonLat> date_ll,
			Double bw,        //Xue: bandwidth in RBF kernel
			Double maxshift,  //Xue: the minimal shift before termination 
			Double cutoff     //Xue: the radius of the circle
			) {
		HashMap<LonLat, Integer> p_count = new HashMap<LonLat, Integer>();
		while(date_ll.size()>0) {
			// choose initial point 
			LonLat init = null;
			Integer z = 0;
			for(String d : date_ll.keySet()) {
				init = date_ll.get(d);
				z+=1;
				if(z==1) {
					break;
				}
			}
			LonLat befmean = init;
			LonLat newmean = new LonLat(0d,0d);
			while(befmean.distance(newmean)>maxshift) {     //Xue: the minimal shift before termination 
				if(newmean.getLon()!=0d) {
					befmean = newmean;
				}
				Double tmplon = 0d;
				Double tmplat = 0d;
				Double tmpwei = 0d;
				for(String d : date_ll.keySet()) {
					LonLat p = date_ll.get(d);
					Double distance = befmean.distance(p);
					if(distance<cutoff) {                  //Xue: the radius of the circle
						Double dist2 = Math.pow(distance, 2d);
						Double wei = Math.exp((dist2)/(-2d*(Math.pow(bw, 2d))));  //Xue: bw, bandwidth in RBF kernel
						tmplon += wei*p.getLon();
						tmplat += wei*p.getLat();
						tmpwei += wei;
					}
				}
				newmean = new LonLat(tmplon/tmpwei, tmplat/tmpwei);
			}
			Integer counter = 0;
			for(Iterator<String> i = date_ll.keySet().iterator();i.hasNext();){
				String k = i.next();
				LonLat p = date_ll.get(k);
				if(p.distance(newmean)<cutoff){
					i.remove();
					counter+=1;
				}
			}
			p_count.put(newmean, counter);
		}
		// now we have the p_count --> get the p with most count
		Integer maxcount = 0;
		LonLat res = new LonLat(0d,0d);
		for(LonLat p : p_count.keySet()) {
			if(p_count.get(p)>maxcount) {
				res = p;
				maxcount = p_count.get(p);
			}
		}
		return res;
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
